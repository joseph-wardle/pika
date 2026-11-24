// Virtual shadow pages, redrawn only when their light or a caster over them changes.

#include "view.slang"

// src/shadow.rs mirrors it.
struct Constants
{
    float4 unproject[4];
    View *camera;
    float *points;
    uint2 *vertices;
    float4 *instances;
    uint *owners;
    float *boxes;
    uint *records;
    Shadow *shadows;
    // Each tile's kept page, then its page this frame (a live copy or the page).
    uint *pages;
    uint *tiles;
    Atomic<uint> *flags;
    // min x, min y, min(across - 1 - x), min(across - 1 - y), so all start at 0xffffffff.
    Atomic<uint> *bounds;
    // 1 - clip depth, 1 where nothing was drawn.
    float *depth;
    Owner *page_owners;
    State *state;
    uint *indirect;
    uint4 *draws;
    Atomic<uint> *pool;
    uint *free;
    uint *drawn;
    uint *copies;
    // 8 words per instance: its box when it last moved, then that frame (0 when fresh).
    float *last;
    uint *moved;
    // shadow, level | face << 8 | flag << 16, then tiles a to b at 8 bits each
    uint4 *walks;
    uint count;
    uint pool_pages;
    uint fresh;
    uint frame;
};

// MARK: a pixel wants it. PARENT: a finer tile is marked. NEED: held without a page.
static const uint MARK = 1;
static const uint PARENT = 2;
static const uint NEED = 4;
static const uint DRAW = 8;
static const uint LIVE = 16;

static const uint STARTED = 0x80000000;
static const uint STOPPED = 0x40000000;
static const uint INSTANCE = 0x3fffffff;

static const uint FREE = 0xffffffff;

// A page's content depends on its tile's place.
struct Owner
{
    uint shadow;
    int2 place;
};

// Zeroed each frame. src/shadow.rs mirrors it.
struct State
{
    Atomic<uint> page_draws;
    // `dropped` is LEVELS for none.
    Atomic<uint> held;
    Atomic<uint> wanted;
    Atomic<uint> dropped;
    Atomic<uint> past;
    // `triangles` wraps past 2^32.
    Atomic<uint> drawn;
    Atomic<uint> triangles;
    Atomic<uint> live_draws;
    Atomic<uint> freed;
    Atomic<uint> taken;
    Atomic<uint> lives;
    Atomic<uint> walked;
    // [top * LEVELS + level], top being the shadow's coarsest level.
    Atomic<uint> marked[LEVELS * LEVELS];
    Atomic<uint> parents[LEVELS * LEVELS];
    Atomic<uint> both[LEVELS];
    // `ranked` is LEVELS for none.
    Atomic<uint> ranked;
    Atomic<uint> room;
};

static const uint CAPACITY = 1 << 19;
static const uint WALKS = 1 << 18;

struct Push
{
    Constants *constants;
    uint first;
    uint count;
    uint stride;
    uint pad;
};

[[vk::push_constant]]
ConstantBuffer<Push> push;

static const uint FACES = 6;

// Apart, so still casters walk only the pages drawn, not the live copies around moving ones.
Atomic<uint> *bounds(uint j, uint i, uint face, bool live)
{
    return push.constants->bounds + 4 * (((j * LEVELS + i) * FACES + face) * 2 + uint(live));
}

uint *pages(uint j)
{
    return push.constants->pages + j * LEVELS * TILEMAP;
}

uint *tiles(uint j)
{
    return push.constants->tiles + j * LEVELS * TILEMAP;
}

Atomic<uint> *flags(uint j)
{
    return push.constants->flags + j * LEVELS * TILEMAP;
}

// Within two texels of q, for the lookup's offset and depth rounding.
bool mark_around(uint j, float2 q, uint i)
{
    Shadow *s = push.constants->shadows + j;
    float margin = 2 * tile_width(s, i) / PAGE;
    int2 lo = tile_at(s, q - margin, i);
    int2 hi = tile_at(s, q + margin, i);
    int2 a = max(lo, 0);
    int2 b = min(hi, int(TILES) - 1);
    for (int y = a.y; y <= b.y; y++)
        for (int x = a.x; x <= b.x; x++)
            flags(j)[tile_index(s, i, 0, int2(x, y))].store(MARK);
    return inside(lo) && inside(hi);
}

// On every face the texels reach, since the lookup's offset can cross an edge.
void mark_faces(uint j, float3 q, uint i)
{
    Shadow *s = push.constants->shadows + j;
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
                flags(j)[tile_index(s, i, face, int2(x, y))].store(MARK);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mark(uint3 thread : SV_DispatchThreadID)
{
    Constants *p = push.constants;
    View *v = p->camera;
    uint2 extent = v->extent;
    uint columns = (extent.x + LIGHT_TILE - 1) / LIGHT_TILE;
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
        uint2 listed = v->light_tiles[pixel.y / LIGHT_TILE * columns + pixel.x / LIGHT_TILE];
        for (uint slot = listed.x; slot < listed.x + listed.y; slot++)
        {
            Light light = v->lights[v->tile_lights[slot]];
            // Light links are the mesh's, which the depth does not keep.
            if (light.shadow == NO_SHADOW || !lit_by(light, P, 0xffffffff))
                continue;
            uint j = light.shadow;
            Shadow *s = p->shadows + j;
            float3 q = to_light(s, P);
            if (!distant(s))
            {
                mark_faces(j, q, lod_at(s, on_face(face_of(q), q).z, w));
                continue;
            }
            uint i = level_at(s, q, w);
            if (!mark_around(j, q.xy, i) && i + 1 < s->levels)
                mark_around(j, q.xy, i + 1);
        }
    }
}

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
    if (distant(s))
        t.at = (t.at - s->origin[t.level]) & int(TILES - 1);
    return t;
}

int2 place_of(Shadow *s, Tile t)
{
    return distant(s) ? t.at + s->origin[t.level] : t.at;
}

uint parent(Shadow *s, Tile t)
{
    uint i = t.level;
    int2 up = ((t.at + s->origin[i]) >> 1) - s->origin[i + 1];
    return tile_index(s, i + 1, t.face, up);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void count(uint3 thread : SV_DispatchThreadID)
{
    Constants *p = push.constants;
    for (uint k = thread.x; k < push.count; k += push.stride)
    {
        uint j = k / (LEVELS * TILEMAP);
        uint e = k % (LEVELS * TILEMAP);
        Shadow *s = p->shadows + j;
        if (e >= start(s, s->levels) || (flags(j)[e].load() & MARK) == 0)
            continue;
        Tile t = tile_of(s, e);
        uint top = s->levels - 1;
        p->state->marked[top * LEVELS + t.level].increment();
        if (t.level == top)
            continue;
        uint before = flags(j)[parent(s, t)].or(PARENT);
        if ((before & PARENT) != 0)
            continue;
        p->state->parents[top * LEVELS + t.level + 1].increment();
        if (t.level + 1 == top && (before & MARK) != 0)
            p->state->both[top].increment();
    }
}

bool wanted(uint flags, uint i, uint top, uint dropped)
{
    flags &= MARK | PARENT;
    if (dropped >= top || i < dropped)
        return (flags & MARK) != 0;
    if (i == dropped)
        return false;
    return i < top ? (flags & PARENT) != 0 : flags != 0;
}

uint wanted_at(uint i, uint top, uint dropped)
{
    Constants *p = push.constants;
    uint marked = p->state->marked[top * LEVELS + i].load();
    if (dropped >= top || i < dropped)
        return marked;
    if (i == dropped)
        return 0;
    uint parents = p->state->parents[top * LEVELS + i].load();
    return i < top ? parents : parents + marked - p->state->both[top].load();
}

// Past the pool, levels from `dropped` up draw their marks into their parents, so far pages
// drop a LOD first. If that still overflows, the level the pool cuts is ranked.
[shader("compute")]
[numthreads(1, 1, 1)]
void choose(uint3 thread : SV_DispatchThreadID)
{
    Constants *p = push.constants;
    uint total = 0;
    uint coarsest = 0;
    for (uint top = 0; top < LEVELS; top++)
    {
        for (uint i = 0; i <= top; i++)
        {
            uint marked = p->state->marked[top * LEVELS + i].load();
            total += marked;
            if (marked > 0)
                coarsest = max(coarsest, top);
        }
    }
    uint dropped = LEVELS;
    for (int first = int(coarsest) - 1; total > p->pool_pages && first >= 0; first--)
    {
        uint held = 0;
        for (uint top = 0; top < LEVELS; top++)
            for (uint i = 0; i <= top; i++)
                held += wanted_at(i, top, first);
        dropped = first;
        if (held <= p->pool_pages)
            break;
    }
    uint base = 0;
    uint ranked = LEVELS;
    uint room = 0;
    for (int i = LEVELS - 1; i >= 0; i--)
    {
        uint level = 0;
        for (uint top = i; top < LEVELS; top++)
            level += wanted_at(i, top, dropped);
        if (ranked == LEVELS && base + level > p->pool_pages)
        {
            ranked = i;
            room = p->pool_pages - base;
        }
        base += level;
    }
    p->state->ranked.store(ranked);
    p->state->room.store(room);
    p->state->held.store(base);
    p->state->wanted.store(total);
    p->state->dropped.store(dropped);
}

void widen(uint j, Tile t, bool live)
{
    Shadow *s = push.constants->shadows + j;
    uint across = tiles_across(s, t.level);
    Atomic<uint> *b = bounds(j, t.level, t.face, live);
    b[0].min(t.at.x);
    b[1].min(t.at.y);
    b[2].min(across - 1 - t.at.x);
    b[3].min(across - 1 - t.at.y);
}

void draw_page(uint j, Tile t, uint page)
{
    push.constants->drawn[push.constants->state->drawn.increment()] = page;
    widen(j, t, false);
}

void paint(uint j, uint e, Tile t, uint page)
{
    Constants *p = push.constants;
    Shadow *s = p->shadows + j;
    p->page_owners[page].shadow = j;
    p->page_owners[page].place = place_of(s, t);
    flags(j)[e].or(DRAW);
    draw_page(j, t, page);
}

void let_go(uint j, uint e)
{
    tiles(j)[e] = NO_PAGE;
    uint page = pages(j)[e];
    if (page == NO_PAGE)
        return;
    push.constants->page_owners[page].shadow = FREE;
    pages(j)[e] = NO_PAGE;
}

void hold(uint j, uint e, Tile t)
{
    Constants *p = push.constants;
    Shadow *s = p->shadows + j;
    uint page = pages(j)[e];
    tiles(j)[e] = page;
    if (page == NO_PAGE)
        flags(j)[e].or(NEED);
    else if (s->reset != 0 || any(p->page_owners[page].place != place_of(s, t)))
        paint(j, e, t, page);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void release(uint3 thread : SV_DispatchThreadID)
{
    Constants *p = push.constants;
    uint dropped = p->state->dropped.load();
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
        if (!wanted(flags(j)[e].load(), t.level, s->levels - 1, dropped))
        {
            let_go(j, e);
            continue;
        }
        uint ranked = p->state->ranked.load();
        if (ranked == LEVELS || t.level > ranked)
            hold(j, e, t);
        else if (t.level < ranked)
            let_go(j, e);
    }
}

groupshared uint sums[64];

uint sum_before(uint lane, uint v, out uint total)
{
    sums[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint d = 1; d < 64; d <<= 1)
    {
        uint before = lane >= d ? sums[lane - d] : 0;
        GroupMemoryBarrierWithGroupSync();
        sums[lane] += before;
        GroupMemoryBarrierWithGroupSync();
    }
    total = sums[63];
    uint before = sums[lane] - v;
    GroupMemoryBarrierWithGroupSync();
    return before;
}

// In shadow and tile order, since thread order would change which tiles keep a page.
[shader("compute")]
[numthreads(64, 1, 1)]
void rank(uint3 lane : SV_GroupThreadID)
{
    Constants *p = push.constants;
    uint i = p->state->ranked.load();
    if (i == LEVELS)
        return;
    uint room = p->state->room.load();
    uint dropped = p->state->dropped.load();
    uint taken = 0;
    for (uint j = 0; j < p->count; j++)
    {
        Shadow *s = p->shadows + j;
        if (i >= s->levels)
            continue;
        for (uint first = start(s, i); first < start(s, i + 1); first += 64)
        {
            uint e = first + lane.x;
            bool held = e < start(s, i + 1) && wanted(flags(j)[e].load(), i, s->levels - 1, dropped);
            uint total;
            uint before = sum_before(lane.x, held ? 1 : 0, total);
            if (held && taken + before < room)
                hold(j, e, tile_of(s, e));
            else if (held)
                let_go(j, e);
            taken += total;
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gather(uint3 thread : SV_DispatchThreadID)
{
    Constants *p = push.constants;
    for (uint k = thread.x; k < push.count; k += push.stride)
    {
        if (p->page_owners[k].shadow == FREE)
            p->free[p->state->freed.increment()] = k;
    }
}

// Ranking kept held tiles within the pool, so there is a free page for each.
[shader("compute")]
[numthreads(64, 1, 1)]
void assign(uint3 thread : SV_DispatchThreadID)
{
    Constants *p = push.constants;
    for (uint k = thread.x; k < push.count; k += push.stride)
    {
        uint j = k / (LEVELS * TILEMAP);
        uint e = k % (LEVELS * TILEMAP);
        if ((flags(j)[e].load() & NEED) == 0)
            continue;
        uint n = p->state->taken.increment();
        if (n >= p->state->freed.load())
            continue;
        uint page = p->free[n];
        pages(j)[e] = page;
        tiles(j)[e] = page;
        paint(j, e, tile_of(p->shadows + j, e), page);
    }
}

uint corners_of(uint instance, out float3 corners[8])
{
    Constants *p = push.constants;
    uint r = p->owners[instance];
    float *box = p->boxes + 6 * r;
    float4 *m = p->instances + 7 * instance;
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
    uint link = asuint(push.constants->instances[7 * instance + 6].z);
    return (link & s->link) == s->link;
}

bool covers(Shadow *s, uint i, uint face, float3 corners[8], out int2 a, out int2 b)
{
    a = 0;
    b = -1;
    if (distant(s))
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

void dirty(uint j, uint i, uint face, int2 at)
{
    Shadow *s = push.constants->shadows + j;
    uint e = tile_index(s, i, face, at);
    uint page = pages(j)[e];
    if (page == NO_PAGE || (flags(j)[e].or(DRAW) & DRAW) != 0)
        return;
    Tile t;
    t.level = i;
    t.face = face;
    t.at = at;
    draw_page(j, t, page);
}

void enliven(uint j, uint i, uint face, int2 at)
{
    Constants *p = push.constants;
    Shadow *s = p->shadows + j;
    uint e = tile_index(s, i, face, at);
    uint page = pages(j)[e];
    if (page == NO_PAGE || (flags(j)[e].or(LIVE) & LIVE) != 0)
        return;
    // A tile has at most one, so they fit the pool's second half.
    uint c = p->state->lives.increment();
    p->copies[c] = page;
    tiles(j)[e] = p->pool_pages + c;
    Tile t;
    t.level = i;
    t.face = face;
    t.at = at;
    widen(j, t, true);
}

void visit(uint j, uint i, uint face, int2 at, uint flag)
{
    if (flag == LIVE)
        enliven(j, i, face, at);
    else
        dirty(j, i, face, at);
}

// A thread per tile, not per box, which would hold its group until the largest box is done.
void walk_tiles(uint j, uint i, uint face, int2 a, int2 b, uint flag)
{
    Constants *p = push.constants;
    uint w = p->state->walked.increment();
    if (w < WALKS)
    {
        uint span = uint(a.x) | uint(a.y) << 8 | uint(b.x) << 16 | uint(b.y) << 24;
        p->walks[w] = uint4(j, i | face << 8 | flag << 16, span, 0);
        return;
    }
    for (int y = a.y; y <= b.y; y++)
    {
        for (int x = a.x; x <= b.x; x++)
            visit(j, i, face, int2(x, y), flag);
    }
}

// A started instance redraws the pages its old box covered, which hold it. A stopped one redraws
// those its box covers now, which don't yet.
[shader("compute")]
[numthreads(64, 1, 1)]
void moving(uint3 thread : SV_DispatchThreadID)
{
    Constants *p = push.constants;
    for (uint k = thread.x; k < push.count; k += push.stride)
    {
        uint j = k % p->count;
        uint entry = p->moved[k / (LEVELS * p->count)];
        uint instance = entry & INSTANCE;
        uint i = k / p->count % LEVELS;
        Shadow *s = p->shadows + j;
        if (i >= s->levels || !casts(instance, s))
            continue;
        float3 now[8];
        corners_of(instance, now);
        float *box = p->last + 8 * instance;
        float3 then[8];
        for (uint c = 0; c < 8; c++)
        {
            bool3 high = (c & uint3(1, 2, 4)) != 0;
            then[c] = select(high, float3(box[3], box[4], box[5]), float3(box[0], box[1], box[2]));
        }
        for (uint face = 0; face < face_count(s); face++)
        {
            int2 a, b;
            if ((entry & STARTED) != 0 && covers(s, i, face, then, a, b))
                walk_tiles(j, i, face, a, b, DRAW);
            if (!covers(s, i, face, now, a, b))
                continue;
            walk_tiles(j, i, face, a, b, (entry & STOPPED) != 0 ? DRAW : LIVE);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void walk(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    Constants *p = push.constants;
    uint count = min(p->state->walked.load(), WALKS);
    for (uint w = group.x; w < count; w += push.stride / 64)
    {
        uint4 k = p->walks[w];
        uint i = k.y & 0xff;
        uint face = k.y >> 8 & 0xff;
        int2 a = int2(k.z & 0xff, k.z >> 8 & 0xff);
        int2 b = int2(k.z >> 16 & 0xff, k.z >> 24);
        uint across = uint(b.x - a.x + 1);
        uint area = across * uint(b.y - a.y + 1);
        for (uint n = lane.x; n < area; n += 64)
            visit(k.x, i, face, a + int2(n % across, n / across), k.y >> 16);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void remember(uint3 thread : SV_DispatchThreadID)
{
    Constants *p = push.constants;
    for (uint k = thread.x; k < push.count; k += push.stride)
    {
        uint instance = p->fresh != 0 ? k : p->moved[k] & INSTANCE;
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
        box[6] = asfloat(p->fresh != 0 ? 0 : p->frame);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void copy(uint3 thread : SV_DispatchThreadID)
{
    Constants *p = push.constants;
    uint texels = p->state->lives.load() * PAGE * PAGE;
    for (uint k = thread.x; k < texels; k += push.stride)
    {
        uint c = k / (PAGE * PAGE);
        uint texel = k % (PAGE * PAGE);
        uint depth = p->pool[p->copies[c] * PAGE * PAGE + texel].load();
        p->pool[(p->pool_pages + c) * PAGE * PAGE + texel].store(depth);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clear(uint3 thread : SV_DispatchThreadID)
{
    Constants *p = push.constants;
    uint texels = p->state->drawn.load() * PAGE * PAGE;
    for (uint k = thread.x; k < texels; k += push.stride)
        p->pool[p->drawn[k / (PAGE * PAGE)] * PAGE * PAGE + k % (PAGE * PAGE)].store(0xffffffff);
}

struct Draw
{
    uint instance;
    uint shadow;
    uint level;
    uint face;
    bool live;
    int2 lo;
    int2 hi;
};

uint4 pack(Draw t)
{
    uint place = t.level | t.face << 8 | uint(t.live) << 16;
    uint span = uint(t.lo.x) | uint(t.lo.y) << 8 | uint(t.hi.x) << 16 | uint(t.hi.y) << 24;
    return uint4(t.instance, t.shadow, place, span);
}

Draw unpack(uint4 w)
{
    Draw t;
    t.instance = w.x;
    t.shadow = w.y;
    t.level = w.z & 0xff;
    t.face = w.z >> 8 & 0xff;
    t.live = (w.z >> 16) != 0;
    t.lo = int2(w.w & 0xff, w.w >> 8 & 0xff);
    t.hi = int2(w.w >> 16 & 0xff, w.w >> 24);
    return t;
}

bool moves(uint instance)
{
    return asuint(push.constants->last[8 * instance + 6]) == push.constants->frame;
}

bool touched(uint j, uint i, uint face, uint instance)
{
    return bounds(j, i, face, moves(instance))[0].load() < TILES;
}

void emit(Draw t, uint r)
{
    Constants *p = push.constants;
    uint d;
    if (t.live)
        d = p->state->live_draws.increment();
    else
        d = p->state->page_draws.increment();
    if (d >= CAPACITY)
    {
        p->state->past.increment();
        return;
    }
    d += t.live ? CAPACITY : 0;
    uint indices = p->records[5 * r];
    p->state->triangles.add(indices / 3);
    p->indirect[5 * d] = indices;
    p->indirect[5 * d + 1] = 1;
    p->indirect[5 * d + 2] = p->records[5 * r + 2];
    p->indirect[5 * d + 3] = p->records[5 * r + 3];
    p->indirect[5 * d + 4] = d;
    p->draws[d] = pack(t);
}

// A draw per rectangle of drawn tiles, so fragments land only in tiles being drawn.
void list(uint j, uint i, uint face, int2 a, int2 b, uint instance, uint r)
{
    Shadow *s = push.constants->shadows + j;
    bool moving = moves(instance);
    Atomic<uint> *bound = bounds(j, i, face, moving);
    int across = int(tiles_across(s, i));
    a = max(a, int2(bound[0].load(), bound[1].load()));
    b = min(b, across - 1 - int2(bound[2].load(), bound[3].load()));
    uint rows[TILES];
    for (int y = a.y; y <= b.y; y++)
    {
        rows[y] = 0;
        for (int x = a.x; x <= b.x; x++)
        {
            uint e = tile_index(s, i, face, int2(x, y));
            if ((flags(j)[e].load() & (moving ? LIVE : DRAW)) != 0)
                rows[y] |= 1u << uint(x - a.x);
        }
    }
    for (int y = a.y; y <= b.y; y++)
    {
        while (rows[y] != 0)
        {
            uint first = firstbitlow(rows[y]);
            uint above = rows[y] + (1u << first);
            uint run = (above & (~above + 1)) - (1u << first);
            int last = y;
            while (last < b.y && (rows[last + 1] & run) == run)
                rows[++last] &= ~run;
            rows[y] &= ~run;
            Draw t;
            t.instance = instance;
            t.shadow = j;
            t.level = i;
            t.face = face;
            t.live = moving;
            t.lo = int2(a.x + int(first), y);
            t.hi = int2(a.x + int(firstbithigh(run)), last);
            emit(t, r);
        }
    }
}

void cast_one(uint instance, uint j, uint i)
{
    Shadow *s = push.constants->shadows + j;
    if (i >= s->levels || !casts(instance, s))
        return;
    uint faces = face_count(s);
    bool any = false;
    for (uint face = 0; face < faces; face++)
        any = any || touched(j, i, face, instance);
    if (!any)
        return;
    float3 corners[8];
    uint r = corners_of(instance, corners);
    for (uint face = 0; face < faces; face++)
    {
        int2 a, b;
        if (touched(j, i, face, instance) && covers(s, i, face, corners, a, b))
            list(j, i, face, a, b, instance, r);
    }
}

// A frame that draws no page lists none, so scene size costs only frames that draw pages.
[shader("compute")]
[numthreads(64, 1, 1)]
void cast(uint3 thread : SV_DispatchThreadID)
{
    Constants *p = push.constants;
    if (p->state->drawn.load() == 0)
        return;
    for (uint k = thread.x; k < push.count; k += push.stride)
    {
        uint instance = k / (LEVELS * p->count);
        if (!moves(instance))
            cast_one(instance, k % p->count, k / p->count % LEVELS);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cast_moving(uint3 thread : SV_DispatchThreadID)
{
    Constants *p = push.constants;
    for (uint k = thread.x; k < push.count; k += push.stride)
    {
        uint instance = p->moved[k / (LEVELS * p->count)] & INSTANCE;
        if (moves(instance))
            cast_one(instance, k % p->count, k / p->count % LEVELS);
    }
}

struct Varyings
{
    float4 position : SV_Position;
    float clip[4] : SV_ClipDistance;
    // Interpolated perspective-correct, so a local light's depth along its face is exact.
    float depth : DEPTH;
    nointerpolation uint draw : DRAW;
};

[shader("vertex")]
Varyings vs(uint vertex : SV_VulkanVertexID, uint draw : SV_VulkanInstanceID)
{
    Constants *p = push.constants;
    Draw t = unpack(p->draws[draw]);
    Shadow *s = p->shadows + t.shadow;
    uint point = p->vertices[vertex].x;
    float4 local = float4(p->points[3 * point], p->points[3 * point + 1], p->points[3 * point + 2], 1);
    float4 *m = p->instances + 7 * t.instance;
    float3 world = float3(dot(m[0], local), dot(m[1], local), dot(m[2], local));
    float3 q = to_light(s, world);
    Varyings o;
    float2 at = q.xy / tile_width(s, t.level) - float2(s->origin[t.level]);
    float w = 1;
    float z = q.z;
    o.depth = q.z;
    if (!distant(s))
    {
        float3 f = on_face(t.face, q);
        at = (f.xy + f.z) * float(tiles_across(s, t.level)) / 2;
        w = f.z;
        z = s->far * (f.z - s->near) / (s->far - s->near);
        o.depth = f.z;
    }
    o.position = float4(at / TILES * 2 - w, z, w);
    float2 low = at - float2(t.lo) * w;
    float2 high = float2(t.hi + 1) * w - at;
    o.clip = { low.x, low.y, high.x, high.y };
    o.draw = draw;
    return o;
}

// The flag is checked again, since a moving caster written into a tile without a live copy would
// stay in its page.
[shader("fragment")]
void fs(Varyings v)
{
    Constants *p = push.constants;
    Draw t = unpack(p->draws[v.draw]);
    Shadow *s = p->shadows + t.shadow;
    uint2 x = uint2(v.position.xy);
    uint e = tile_index(s, t.level, t.face, int2(x / PAGE));
    if ((flags(t.shadow)[e].load() & (t.live ? LIVE : DRAW)) == 0)
        return;
    uint page = t.live ? tiles(t.shadow)[e] : pages(t.shadow)[e];
    uint2 texel = x % PAGE;
    // Positive float bits order as integers do.
    p->pool[(page * PAGE + texel.y) * PAGE + texel.x].min(asuint(v.depth));
}
