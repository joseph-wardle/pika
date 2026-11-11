// Virtual shadow pages, in the frame's order: mark the tiles the camera's pixels read, count
// them, choose which levels the pool holds, release the pages of tiles no longer held, gather
// the free pages, assign them to the tiles that need one, find the held pages a moving
// caster's old or new box covers, remember each moved caster's box, clear the pages to draw,
// list each caster instance once per page to draw that it meets, and draw that list by
// atomic min, so the pool needs no depth attachment. A page held and unchanged is not drawn.

#include "view.slang"

// src/shadow.rs Pages, written once a frame.
struct Pages
{
    // Camera clip space to world: rows of the column-vector matrix.
    float4 unproject[4];
    View *camera;
    // Pulls the casters' vertices; its records give each mesh's draw.
    View *casters;
    Shadow *shadows;
    // Every shadow's tiles, as each Shadow's tiles points into them, kept from frame to frame.
    uint *tiles;
    // Per tile, this frame's flags: MARK to PAINTED.
    Atomic<uint> *marks;
    // Per shadow, level and face, the bounds of the tiles drawn this frame: least x and y,
    // then least across - 1 - x and y, so all four start at 0xffffffff and fall by min.
    Atomic<uint> *bounds;
    // The camera's depth AOV: 1 - clip depth, 1 where nothing was drawn.
    float *depth;
    // Per pool page: its shadow, its tile in that shadow's tiles and the tile's place (a
    // distant light's from its light-space origin); FREE in x when no tile holds it.
    uint4 *owners;
    // State: see STATE.
    Atomic<uint> *state;
    // vk::DrawIndexedIndirectCommand, 5 words each, and per draw its instance and page.
    uint *draws;
    uint2 *pairs;
    Atomic<uint> *pool;
    // The pages free once tiles let theirs go, and the pages drawn this frame.
    uint *free;
    uint *drawn;
    // Per instance, 8 words: its world box when it last moved, low corner then high, and
    // the frame it last moved.
    float *last;
    // The instances whose transform or points moved this frame.
    uint *moved;
    uint count;
    // Pool pages, and draws the lists hold.
    uint pages;
    uint capacity;
    // 1 when the instances' last boxes are not theirs: every page is drawn anew.
    uint fresh;
    uint frame;
    uint pad;
};

// A tile's flags: marked by a pixel, a finer mark's parent, held but without a page, to be
// drawn this frame, and drawn because its page is new or shows a changed light, not because a
// caster moved.
static const uint MARK = 1;
static const uint PARENT = 2;
static const uint NEED = 4;
static const uint DRAW = 8;
static const uint PAINTED = 16;

static const uint FREE = 0xffffffff;

// The state's words (src/shadow.rs STATE): draws listed, pages wanted once levels drop,
// pages marked, the first level dropped (LEVELS for none), draws past the capacity, pages
// drawn, triangles drawn, and of those the triangles of casters that did not move drawn into
// pages that only a moving caster changed (both wrap past 2^32); the free pages and those
// taken; then, by the coarsest level of the shadows counted and by level, tiles marked and
// tiles that are a finer mark's parent; by coarsest level, its tiles both; and by level,
// pages ranked and where the level's ranks start.
static const uint DRAWN = 5;
static const uint TRIANGLES = 6;
static const uint STILL = 7;
static const uint FREED = 8;
static const uint TAKEN = 9;
static const uint MARKED = 10;
static const uint PARENTS = MARKED + LEVELS * LEVELS;
static const uint BOTH = PARENTS + LEVELS * LEVELS;
static const uint NEXT = BOTH + LEVELS;
static const uint BASE = NEXT + LEVELS;

struct Push
{
    Pages *pages;
    uint first;
    uint count;
    uint stride;
    uint pad;
};

[[vk::push_constant]]
ConstantBuffer<Push> push;

// Faces' bounds per shadow and level.
static const uint FACES = 6;

Atomic<uint> *bounds(uint j, uint i, uint face)
{
    return push.pages->bounds + 4 * ((j * LEVELS + i) * FACES + face);
}

// Shadow j's tiles and their flags.
uint *entries(uint j)
{
    return push.pages->tiles + j * LEVELS * TILEMAP;
}

Atomic<uint> *marks(uint j)
{
    return push.pages->marks + j * LEVELS * TILEMAP;
}

// Marks the tiles of a distant light's level i within two of its texels of q: one for the
// lookup's offset off the surface, one for the depth's rounding. True when they all lie
// inside the level.
bool mark_around(uint j, float2 q, uint i)
{
    Shadow *s = push.pages->shadows + j;
    float margin = 2 * tile_width(s, i) / PAGE;
    int2 lo = tile_at(s, q - margin, i);
    int2 hi = tile_at(s, q + margin, i);
    int2 a = max(lo, 0);
    int2 b = min(hi, int(TILES) - 1);
    for (int y = a.y; y <= b.y; y++)
        for (int x = a.x; x <= b.x; x++)
            marks(j)[tile_index(s, i, 0, int2(x, y))].store(MARK);
    return inside(lo) && inside(hi);
}

// Marks a local light's tiles at LOD i within two of its texels of q, on every face they
// reach, since the lookup's offset may carry P across a face's edge.
void mark_faces(uint j, float3 q, uint i)
{
    Shadow *s = push.pages->shadows + j;
    uint n = tiles_across(s, i);
    float margin = 2 * 2.0 / (n * PAGE);
    for (uint face = 0; face < s->faces; face++)
    {
        float3 f = on_face(face, q);
        if (f.z <= 0)
            continue;
        float2 uv = f.xy / f.z;
        if (any(uv - margin > 1) || any(uv + margin < -1))
            continue;
        int2 a = clamp(int2(floor(face_tiles(uv - margin, n))), 0, int(n) - 1);
        int2 b = clamp(int2(floor(face_tiles(uv + margin, n))), 0, int(n) - 1);
        for (int y = a.y; y <= b.y; y++)
            for (int x = a.x; x <= b.x; x++)
                marks(j)[tile_index(s, i, face, int2(x, y))].store(MARK);
    }
}

// Per pixel, over the lights its screen tile lists, as shading reads them.
[shader("compute")]
[numthreads(64, 1, 1)]
void mark(uint3 thread : SV_DispatchThreadID)
{
    Pages *p = push.pages;
    View *v = p->camera;
    uint2 extent = v->extent;
    uint columns = (extent.x + TILE - 1) / TILE;
    for (uint k = thread.x; k < push.count; k += push.stride)
    {
        float d = p->depth[k];
        if (d >= 1)
            continue;
        uint2 pixel = uint2(k % extent.x, k / extent.x);
        float2 xy = (float2(pixel) + 0.5) / float2(extent) * 2 - 1;
        float4 c = float4(xy, 1 - d, 1);
        float4 h = float4(dot(p->unproject[0], c), dot(p->unproject[1], c),
                          dot(p->unproject[2], c), dot(p->unproject[3], c));
        float3 P = h.xyz / h.w;
        float w = dot(v->view_projection[3], float4(P, 1));
        uint2 listed = v->tiles[pixel.y / TILE * columns + pixel.x / TILE];
        for (uint slot = listed.x; slot < listed.x + listed.y; slot++)
        {
            Light light = v->lights[v->tile_lights[slot]];
            // Light links are the mesh's, which the depth does not keep.
            if (light.shadow == NO_SHADOW || !lit_by(light, P, 0xffffffff))
                continue;
            uint j = light.shadow;
            Shadow *s = p->shadows + j;
            float3 q = to_light(s, P);
            if (s->faces > 0)
            {
                mark_faces(j, q, lod_at(s, on_face(face_of(q), q).z, w));
                continue;
            }
            uint i = level_at(s, q, w);
            // A margin past the level's edge may send the lookup a level up.
            if (!mark_around(j, q.xy, i) && i + 1 < s->levels)
                mark_around(j, q.xy, i + 1);
        }
    }
}

// Shadow j's tile e: its level, face and place, a distant light's from its level's origin.
struct Tile
{
    uint level;
    uint face;
    int2 at;
};

Tile tile_of(Shadow *s, uint e)
{
    Tile t;
    t.level = 0;
    while (t.level + 1 < s->levels && start(s, t.level + 1) <= e)
        t.level++;
    uint n = tiles_across(s, t.level);
    uint r = e - start(s, t.level);
    t.face = r / (n * n);
    t.at = int2(r % n, r / n % n);
    // Unwrapped; tile_index wraps it again.
    if (s->faces == 0)
        t.at = (t.at - s->origin[t.level]) & int(TILES - 1);
    return t;
}

// Where a tile lies from its light-space origin, which a page's content depends on.
int2 place_of(Shadow *s, Tile t)
{
    return s->faces == 0 ? t.at + s->origin[t.level] : t.at;
}

// The level i + 1 tile holding level i's tile t; a local light's LODs have no origin.
uint parent(Shadow *s, Tile t)
{
    uint i = t.level;
    int2 up = ((t.at + s->origin[i]) >> 1) - s->origin[i + 1];
    return tile_index(s, i + 1, t.face, up);
}

// Over every shadow's tiles: counts the marks, flags each mark's parent and counts each
// parent once. Shadows are counted by their coarsest level, which has no parent.
[shader("compute")]
[numthreads(64, 1, 1)]
void count(uint3 thread : SV_DispatchThreadID)
{
    Pages *p = push.pages;
    for (uint k = thread.x; k < push.count; k += push.stride)
    {
        uint j = k / (LEVELS * TILEMAP);
        uint e = k % (LEVELS * TILEMAP);
        Shadow *s = p->shadows + j;
        if (e >= start(s, s->levels) || (marks(j)[e].load() & MARK) == 0)
            continue;
        Tile t = tile_of(s, e);
        uint top = s->levels - 1;
        p->state[MARKED + top * LEVELS + t.level].increment();
        if (t.level == top)
            continue;
        uint before = marks(j)[parent(s, t)].or(PARENT);
        if ((before & PARENT) != 0)
            continue;
        p->state[PARENTS + top * LEVELS + t.level + 1].increment();
        if (t.level + 1 == top && (before & MARK) != 0)
            p->state[BOTH + top].increment();
    }
}

// Whether a tile at level i holding flags is held when levels from dropped up draw their
// marks into their parents; top is its shadow's coarsest level, which cannot drop.
bool wanted(uint flags, uint i, uint top, uint dropped)
{
    flags &= MARK | PARENT;
    if (dropped >= top || i < dropped)
        return (flags & MARK) != 0;
    if (i == dropped)
        return false;
    return i < top ? (flags & PARENT) != 0 : flags != 0;
}

// Tiles level i holds, over shadows whose coarsest level is top, when dropped drops.
uint wanted_at(uint i, uint top, uint dropped)
{
    Pages *p = push.pages;
    uint marked = p->state[MARKED + top * LEVELS + i].load();
    if (dropped >= top || i < dropped)
        return marked;
    if (i == dropped)
        return 0;
    uint parents = p->state[PARENTS + top * LEVELS + i].load();
    return i < top ? parents : parents + marked - p->state[BOTH + top].load();
}

// One thread. Past the pool, every shadow's levels from the first dropped one up draw their
// marks into their parents instead, so the farthest pages drop a LOD first; lookups find the
// parents by walking up. Tiles rank coarsest first: should even dropping every level
// overflow, the finest go.
[shader("compute")]
[numthreads(1, 1, 1)]
void choose(uint3 thread : SV_DispatchThreadID)
{
    Pages *p = push.pages;
    uint total = 0;
    uint coarsest = 0;
    for (uint top = 0; top < LEVELS; top++)
    {
        for (uint i = 0; i <= top; i++)
        {
            uint marked = p->state[MARKED + top * LEVELS + i].load();
            total += marked;
            if (marked > 0)
                coarsest = max(coarsest, top);
        }
    }
    uint dropped = LEVELS;
    for (int first = int(coarsest) - 1; total > p->pages && first >= 0; first--)
    {
        uint held = 0;
        for (uint top = 0; top < LEVELS; top++)
            for (uint i = 0; i <= top; i++)
                held += wanted_at(i, top, first);
        dropped = first;
        if (held <= p->pages)
            break;
    }
    uint base = 0;
    for (int i = LEVELS - 1; i >= 0; i--)
    {
        p->state[BASE + i].store(base);
        for (uint top = i; top < LEVELS; top++)
            base += wanted_at(i, top, dropped);
    }
    p->state[1].store(base);
    p->state[2].store(total);
    p->state[3].store(dropped);
}

// Lists shadow j's page for tile t to be cleared and drawn this frame.
void draw_page(uint j, Tile t, uint page)
{
    Pages *p = push.pages;
    Shadow *s = p->shadows + j;
    p->drawn[p->state[DRAWN].increment()] = page;
    uint across = tiles_across(s, t.level);
    Atomic<uint> *b = bounds(j, t.level, t.face);
    b[0].min(t.at.x);
    b[1].min(t.at.y);
    b[2].min(across - 1 - t.at.x);
    b[3].min(across - 1 - t.at.y);
}

// Gives shadow j's tile e, t, the page, drawn anew.
void paint(uint j, uint e, Tile t, uint page)
{
    Pages *p = push.pages;
    Shadow *s = p->shadows + j;
    p->owners[page] = uint4(j, e, asuint(place_of(s, t)));
    marks(j)[e].or(DRAW | PAINTED);
    draw_page(j, t, page);
}

// Shadow j's tile e lets its page go, if it holds one.
void let_go(uint j, uint e)
{
    uint page = entries(j)[e];
    if (page == NO_PAGE)
        return;
    push.pages->owners[page].x = FREE;
    entries(j)[e] = NO_PAGE;
}

// Over every shadow's tiles: ranks the tiles held this frame by level, coarsest first, and
// those ranked past the pool are not held. A tile not held lets its page go; one held keeps
// its page, drawn anew should its light or place have changed, or needs one.
[shader("compute")]
[numthreads(64, 1, 1)]
void release(uint3 thread : SV_DispatchThreadID)
{
    Pages *p = push.pages;
    uint dropped = p->state[3].load();
    for (uint k = thread.x; k < push.count; k += push.stride)
    {
        uint j = k / (LEVELS * TILEMAP);
        uint e = k % (LEVELS * TILEMAP);
        Shadow *s = p->shadows + j;
        if (e >= start(s, s->levels))
        {
            let_go(j, e);
            continue;
        }
        Tile t = tile_of(s, e);
        if (!wanted(marks(j)[e].load(), t.level, s->levels - 1, dropped))
        {
            let_go(j, e);
            continue;
        }
        uint rank = p->state[BASE + t.level].load() + p->state[NEXT + t.level].increment();
        if (rank >= p->pages)
        {
            let_go(j, e);
            continue;
        }
        uint page = entries(j)[e];
        if (page == NO_PAGE)
            marks(j)[e].or(NEED);
        else if (s->reset != 0 || any(asint(p->owners[page].zw) != place_of(s, t)))
            paint(j, e, t, page);
    }
}

// Over every pool page: lists the free ones.
[shader("compute")]
[numthreads(64, 1, 1)]
void gather(uint3 thread : SV_DispatchThreadID)
{
    Pages *p = push.pages;
    for (uint k = thread.x; k < push.count; k += push.stride)
    {
        if (p->owners[k].x == FREE)
            p->free[p->state[FREED].increment()] = k;
    }
}

// Over every shadow's tiles: gives each that needs a page a free one. Ranking kept the held
// tiles within the pool, so there is one for each.
[shader("compute")]
[numthreads(64, 1, 1)]
void assign(uint3 thread : SV_DispatchThreadID)
{
    Pages *p = push.pages;
    for (uint k = thread.x; k < push.count; k += push.stride)
    {
        uint j = k / (LEVELS * TILEMAP);
        uint e = k % (LEVELS * TILEMAP);
        if ((marks(j)[e].load() & NEED) == 0)
            continue;
        uint n = p->state[TAKEN].increment();
        if (n >= p->state[FREED].load())
            continue;
        uint page = p->free[n];
        entries(j)[e] = page;
        paint(j, e, tile_of(p->shadows + j, e), page);
    }
}

// The 8 corners of instance's box in world space, and the record it draws.
uint corners_of(uint instance, out float3 corners[8])
{
    View *v = push.pages->casters;
    uint r = v->owners[instance];
    float *box = v->boxes + 6 * r;
    float4 *m = v->instances + 7 * instance;
    for (uint c = 0; c < 8; c++)
    {
        bool3 high = (c & uint3(1, 2, 4)) != 0;
        float4 corner = float4(select(high, float3(box[3], box[4], box[5]),
                                      float3(box[0], box[1], box[2])), 1);
        corners[c] = float3(dot(m[0], corner), dot(m[1], corner), dot(m[2], corner));
    }
    return r;
}

bool casts(uint instance, Shadow *s)
{
    uint link = asuint(push.pages->casters->instances[7 * instance + 6].z);
    return (link & s->link) == s->link;
}

// The tiles a to b of s's level i and face that a box, by its corners, may cover; false when
// it covers none.
bool covers(Shadow *s, uint i, uint face, float3 corners[8], out int2 a, out int2 b)
{
    a = 0;
    b = -1;
    if (s->faces == 0)
    {
        float2 lo = float2(1e38);
        float2 hi = float2(-1e38);
        for (uint c = 0; c < 8; c++)
        {
            float2 q = to_light(s, corners[c]).xy;
            lo = min(lo, q);
            hi = max(hi, q);
        }
        a = max(tile_at(s, lo, i), 0);
        b = min(tile_at(s, hi, i), int(TILES) - 1);
        return all(a <= b);
    }
    float3 q[8];
    float3 near = float3(1e38);
    float3 far = float3(-1e38);
    for (uint c = 0; c < 8; c++)
    {
        q[c] = to_light(s, corners[c]);
        near = min(near, q[c]);
        far = max(far, q[c]);
    }
    // The box's nearest point to the light's centre, past its reach.
    float3 gap = max(max(near, -far), 0);
    if (dot(gap, gap) >= s->far * s->far)
        return false;
    float2 lo = float2(1e38);
    float2 hi = float2(-1e38);
    float deepest = -1e38;
    bool behind = false;
    for (uint c = 0; c < 8; c++)
    {
        float3 f = on_face(face, q[c]);
        deepest = max(deepest, f.z);
        behind = behind || f.z <= 0;
        lo = min(lo, f.xy / f.z);
        hi = max(hi, f.xy / f.z);
    }
    if (deepest <= s->near)
        return false;
    // A box reaching behind the face's plane may cover any of it.
    if (behind)
    {
        lo = -1;
        hi = 1;
    }
    if (any(lo > 1) || any(hi < -1))
        return false;
    int n = int(tiles_across(s, i));
    a = clamp(int2(floor(face_tiles(lo, n))), 0, n - 1);
    b = clamp(int2(floor(face_tiles(hi, n))), 0, n - 1);
    return true;
}

// Lists shadow j's held pages from tile a to b of level i and face to be drawn.
void dirty(uint j, uint i, uint face, int2 a, int2 b)
{
    Shadow *s = push.pages->shadows + j;
    for (int y = a.y; y <= b.y; y++)
    {
        for (int x = a.x; x <= b.x; x++)
        {
            uint e = tile_index(s, i, face, int2(x, y));
            uint page = entries(j)[e];
            if (page == NO_PAGE || (marks(j)[e].or(DRAW) & DRAW) != 0)
                continue;
            Tile t;
            t.level = i;
            t.face = face;
            t.at = int2(x, y);
            draw_page(j, t, page);
        }
    }
}

// Per moved instance and level: the held pages its box covered when it last moved, and
// those it covers now, are drawn anew.
[shader("compute")]
[numthreads(64, 1, 1)]
void moving(uint3 thread : SV_DispatchThreadID)
{
    Pages *p = push.pages;
    for (uint k = thread.x; k < push.count; k += push.stride)
    {
        uint instance = p->moved[k / LEVELS];
        uint i = k % LEVELS;
        float3 now[8];
        corners_of(instance, now);
        float *box = p->last + 8 * instance;
        float3 then[8];
        for (uint c = 0; c < 8; c++)
        {
            bool3 high = (c & uint3(1, 2, 4)) != 0;
            then[c] = select(high, float3(box[3], box[4], box[5]), float3(box[0], box[1], box[2]));
        }
        for (uint j = 0; j < p->count; j++)
        {
            Shadow *s = p->shadows + j;
            if (i >= s->levels || !casts(instance, s))
                continue;
            for (uint face = 0; face < max(s->faces, 1); face++)
            {
                int2 a, b;
                if (covers(s, i, face, then, a, b))
                    dirty(j, i, face, a, b);
                if (covers(s, i, face, now, a, b))
                    dirty(j, i, face, a, b);
            }
        }
    }
}

// Per moved instance, or every one when fresh: its world box and this frame, for the next
// frame it moves in.
[shader("compute")]
[numthreads(64, 1, 1)]
void remember(uint3 thread : SV_DispatchThreadID)
{
    Pages *p = push.pages;
    for (uint k = thread.x; k < push.count; k += push.stride)
    {
        uint instance = p->fresh != 0 ? k : p->moved[k];
        float3 corners[8];
        corners_of(instance, corners);
        float3 lo = corners[0];
        float3 hi = corners[0];
        for (uint c = 1; c < 8; c++)
        {
            lo = min(lo, corners[c]);
            hi = max(hi, corners[c]);
        }
        float *box = p->last + 8 * instance;
        box[0] = lo.x;
        box[1] = lo.y;
        box[2] = lo.z;
        box[3] = hi.x;
        box[4] = hi.y;
        box[5] = hi.z;
        box[6] = asfloat(p->frame);
    }
}

// Over the texels of the pages drawn this frame.
[shader("compute")]
[numthreads(64, 1, 1)]
void clear(uint3 thread : SV_DispatchThreadID)
{
    Pages *p = push.pages;
    uint texels = p->state[DRAWN].load() * PAGE * PAGE;
    for (uint k = thread.x; k < texels; k += push.stride)
        p->pool[p->drawn[k / (PAGE * PAGE)] * PAGE * PAGE + k % (PAGE * PAGE)].store(0xffffffff);
}

// Lists one draw of the instance into each page to draw of shadow j's level i and face its
// tiles from a to b meet.
void list(uint j, uint i, uint face, int2 a, int2 b, uint instance, uint r)
{
    Pages *p = push.pages;
    View *v = p->casters;
    Shadow *s = p->shadows + j;
    Atomic<uint> *bound = bounds(j, i, face);
    // Untouched, the level draws no page there.
    if (bound[0].load() >= TILES)
        return;
    int across = int(tiles_across(s, i));
    a = max(a, int2(bound[0].load(), bound[1].load()));
    b = min(b, across - 1 - int2(bound[2].load(), bound[3].load()));
    bool still = asuint(p->last[8 * instance + 6]) != p->frame;
    for (int y = a.y; y <= b.y; y++)
    {
        for (int x = a.x; x <= b.x; x++)
        {
            uint e = tile_index(s, i, face, int2(x, y));
            uint page = entries(j)[e];
            uint flags = marks(j)[e].load();
            if (page == NO_PAGE || (flags & DRAW) == 0)
                continue;
            uint d = p->state[0].increment();
            if (d >= p->capacity)
            {
                p->state[4].increment();
                continue;
            }
            uint indices = v->records[5 * r].load();
            p->state[TRIANGLES].add(indices / 3);
            if (still && (flags & PAINTED) == 0)
                p->state[STILL].add(indices / 3);
            p->draws[5 * d] = indices;
            p->draws[5 * d + 1] = 1;
            p->draws[5 * d + 2] = v->records[5 * r + 2].load();
            p->draws[5 * d + 3] = v->records[5 * r + 3].load();
            p->draws[5 * d + 4] = d;
            p->pairs[d] = uint2(instance, page);
        }
    }
}

// Per caster instance and level: one draw for each page to draw its box meets.
[shader("compute")]
[numthreads(64, 1, 1)]
void cast(uint3 thread : SV_DispatchThreadID)
{
    Pages *p = push.pages;
    for (uint k = thread.x; k < push.count; k += push.stride)
    {
        uint instance = k / LEVELS;
        uint i = k % LEVELS;
        float3 corners[8];
        uint r = corners_of(instance, corners);
        for (uint j = 0; j < p->count; j++)
        {
            Shadow *s = p->shadows + j;
            if (i >= s->levels || !casts(instance, s))
                continue;
            for (uint face = 0; face < max(s->faces, 1); face++)
            {
                int2 a, b;
                if (covers(s, i, face, corners, a, b))
                    list(j, i, face, a, b, instance, r);
            }
        }
    }
}

struct Varyings
{
    float4 position : SV_Position;
    // Interpolated perspective-correct, so a local light's depth along its face is exact.
    float depth : DEPTH;
    nointerpolation uint page : PAGE;
    uint layer : SV_RenderTargetArrayIndex;
};

// Each draw is one instance into one page, the page's layer, its tile filling clip space.
[shader("vertex")]
Varyings vs(uint vertex : SV_VulkanVertexID, uint draw : SV_VulkanInstanceID)
{
    Pages *p = push.pages;
    View *v = p->casters;
    uint2 pair = p->pairs[draw];
    uint4 owner = p->owners[pair.y];
    Shadow *s = p->shadows + owner.x;
    Tile tile = tile_of(s, owner.y);
    uint point = v->vertices[vertex].x;
    float4 local = float4(v->points[3 * point], v->points[3 * point + 1], v->points[3 * point + 2], 1);
    float4 *m = v->instances + 7 * pair.x;
    float3 world = float3(dot(m[0], local), dot(m[1], local), dot(m[2], local));
    float3 q = to_light(s, world);
    Varyings o;
    if (s->faces == 0)
    {
        int2 t = asint(owner.zw);
        o.position = float4((q.xy / tile_width(s, tile.level) - float2(t)) * 2 - 1, q.z, 1);
        o.depth = q.z;
    }
    else
    {
        // Across the face, (f.xy / f.z + 1) × n - 2t - 1 spans the tile; clip z runs from
        // near to far.
        float3 f = on_face(tile.face, q);
        float n = float(tiles_across(s, tile.level));
        float2 xy = (f.xy + f.z) * n - (2 * float2(tile.at) + 1) * f.z;
        float z = s->far * (f.z - s->near) / (s->far - s->near);
        o.position = float4(xy, z, f.z);
        o.depth = f.z;
    }
    o.page = pair.y;
    o.layer = pair.y;
    return o;
}

[shader("fragment")]
void fs(Varyings v)
{
    uint2 x = uint2(v.position.xy);
    // Depth is positive, where float bits order as integers do.
    push.pages->pool[(v.page * PAGE + x.y) * PAGE + x.x].min(asuint(v.depth));
}
