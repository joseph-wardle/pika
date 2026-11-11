// Virtual shadow pages, in the frame's order: mark the tiles the camera's pixels read,
// allocate pool pages to them, clear those pages, list each caster instance once per page
// it meets, and draw that list by atomic min, so the pool needs no depth attachment.

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
    // Every shadow's tiles, as each Shadow's tiles points into them.
    Atomic<uint> *tiles;
    // Per shadow and level, the held tiles' bounds: least x and y, then least 31 - x and
    // 31 - y, so all four start at 0xffffffff and fall by min.
    Atomic<uint> *bounds;
    // The camera's depth AOV: 1 - clip depth, 1 where nothing was drawn.
    float *depth;
    // Per pool page: its shadow and its tile in that shadow's tiles.
    uint2 *owners;
    // Draws listed, pages held, pages marked, the first level dropped (LEVELS for none),
    // pairs past the capacity.
    Atomic<uint> *state;
    // vk::DrawIndexedIndirectCommand, 5 words each, and per draw its instance and page.
    uint *draws;
    uint2 *pairs;
    Atomic<uint> *pool;
    uint count;
    // Pool pages, and draws the lists hold.
    uint pages;
    uint capacity;
    uint pad;
};

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

static const uint TILEMAP = TILES * TILES;

// Shadow j's tile t at level i.
Atomic<uint> *entry(uint j, uint i, int2 t)
{
    return push.pages->tiles + ((j * LEVELS + i) * TILES + t.y) * TILES + t.x;
}

// Marks with bit 1 the tiles at level i within two of its texels of q: one for the
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
            entry(j, i, int2(x, y))->store(1);
    return inside(lo) && inside(hi);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mark(uint3 thread : SV_DispatchThreadID)
{
    Pages *p = push.pages;
    uint2 extent = p->camera->extent;
    for (uint k = thread.x; k < push.count; k += push.stride)
    {
        float d = p->depth[k];
        if (d >= 1)
            continue;
        float2 xy = (float2(k % extent.x, k / extent.x) + 0.5) / float2(extent) * 2 - 1;
        float4 c = float4(xy, 1 - d, 1);
        float4 h = float4(dot(p->unproject[0], c), dot(p->unproject[1], c),
                          dot(p->unproject[2], c), dot(p->unproject[3], c));
        float3 P = h.xyz / h.w;
        float w = dot(p->camera->view_projection[3], float4(P, 1));
        for (uint j = 0; j < p->count; j++)
        {
            Shadow *s = p->shadows + j;
            float3 q = to_light(s, P);
            uint i = level_at(s, q, w);
            // A margin past the level's edge may send the lookup a level up.
            if (!mark_around(j, q.xy, i) && i + 1 < s->levels)
                mark_around(j, q.xy, i + 1);
        }
    }
}

groupshared uint marked[LEVELS];
// Per level, tiles that are the parent of a mark one level finer.
groupshared uint parents[LEVELS];
// Marks at the coarsest level that are no finer mark's parent.
groupshared uint coarsest;
groupshared uint dropped;
groupshared uint next;

// The level i + 1 tile holding level i's tile t.
int2 parent(Shadow *s, uint i, int2 t)
{
    return ((t + s->origin[i]) >> 1) - s->origin[i + 1];
}

// One group. Every shadow has the same levels (src/shadow.rs). Past the pool, levels from
// the first dropped one up draw their marks into their parents instead, so the farthest
// pages drop a LOD first; lookups find the parents by walking up.
[shader("compute")]
[numthreads(64, 1, 1)]
void allocate(uint3 thread : SV_GroupThreadID)
{
    Pages *p = push.pages;
    uint t = thread.x;
    uint levels = p->shadows[0].levels;
    uint tiles = p->count * LEVELS * TILEMAP;
    if (t < LEVELS)
    {
        marked[t] = 0;
        parents[t] = 0;
    }
    if (t == 0)
    {
        coarsest = 0;
        next = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint k = t; k < tiles; k += 64)
    {
        uint j = k / (LEVELS * TILEMAP);
        uint i = k / TILEMAP % LEVELS;
        int2 at = int2(k % TILES, k / TILES % TILES);
        if (i >= levels || (entry(j, i, at)->load() & 1) == 0)
            continue;
        InterlockedAdd(marked[i], 1);
        if (i + 1 < levels)
            entry(j, i + 1, parent(p->shadows + j, i, at))->or(2);
    }
    AllMemoryBarrierWithGroupSync();
    for (uint k = t; k < tiles; k += 64)
    {
        uint j = k / (LEVELS * TILEMAP);
        uint i = k / TILEMAP % LEVELS;
        int2 at = int2(k % TILES, k / TILES % TILES);
        if (i >= levels)
            continue;
        uint e = entry(j, i, at)->load();
        if ((e & 2) != 0)
            InterlockedAdd(parents[i], 1);
        if (i + 1 == levels && e == 1)
            InterlockedAdd(coarsest, 1);
    }
    GroupMemoryBarrierWithGroupSync();
    if (t == 0)
    {
        uint total = 0;
        for (uint i = 0; i < levels; i++)
            total += marked[i];
        dropped = LEVELS;
        // The coarsest level has no parents to drop to.
        for (int first = int(levels) - 2; total > p->pages && first >= 0; first--)
        {
            uint held = coarsest;
            for (uint i = 0; i < uint(first); i++)
                held += marked[i];
            for (uint i = first + 1; i < levels; i++)
                held += parents[i];
            dropped = first;
            if (held <= p->pages)
                break;
        }
        p->state[2].store(total);
        p->state[3].store(dropped);
    }
    GroupMemoryBarrierWithGroupSync();
    // Coarsest first: should even dropping every level overflow, the finest pages go.
    for (int i = int(levels) - 1; i >= 0; i--)
    {
        for (uint k = t; k < p->count * TILEMAP; k += 64)
        {
            int2 at = int2(k % TILES, k / TILES % TILES);
            Atomic<uint> *e = entry(k / TILEMAP, i, at);
            uint flags = e->load();
            bool wanted = uint(i) < dropped ? (flags & 1) != 0
                          : uint(i) == dropped ? false
                                               : (flags & 2) != 0 || (i + 1 == levels && flags != 0);
            uint page = NO_PAGE;
            if (wanted)
            {
                uint n;
                InterlockedAdd(next, 1, n);
                if (n < p->pages)
                {
                    page = n;
                    uint j = k / TILEMAP;
                    p->owners[page] = uint2(j, i * TILEMAP + at.y * TILES + at.x);
                    Atomic<uint> *b = p->bounds + 4 * (j * LEVELS + i);
                    b[0].min(at.x);
                    b[1].min(at.y);
                    b[2].min(TILES - 1 - at.x);
                    b[3].min(TILES - 1 - at.y);
                }
            }
            e->store(page);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0)
        p->state[1].store(min(next, p->pages));
}

// Over every pool texel; pages past those held are left as they are.
[shader("compute")]
[numthreads(64, 1, 1)]
void clear(uint3 thread : SV_DispatchThreadID)
{
    Pages *p = push.pages;
    uint held = p->state[1].load();
    for (uint k = thread.x; k < push.count && k / (PAGE * PAGE) < held; k += push.stride)
        p->pool[k].store(0xffffffff);
}

// Per caster instance and level: one draw for each held page its box meets. Every shadow
// has the same levels.
[shader("compute")]
[numthreads(64, 1, 1)]
void cast(uint3 thread : SV_DispatchThreadID)
{
    Pages *p = push.pages;
    View *v = p->casters;
    uint levels = p->shadows[0].levels;
    for (uint k = thread.x; k < push.count; k += push.stride)
    {
        uint instance = k / levels;
        uint i = k % levels;
        uint r = v->owners[instance];
        float *box = v->boxes + 6 * r;
        float4 *m = v->instances + 7 * instance;
        uint link = asuint(m[6].z);
        for (uint j = 0; j < p->count; j++)
        {
            Shadow *s = p->shadows + j;
            if ((link & s->link) != s->link)
                continue;
            float2 lo = float2(1e38);
            float2 hi = float2(-1e38);
            for (uint c = 0; c < 8; c++)
            {
                bool3 high = (c & uint3(1, 2, 4)) != 0;
                float4 corner = float4(select(high, float3(box[3], box[4], box[5]),
                                              float3(box[0], box[1], box[2])), 1);
                float3 world = float3(dot(m[0], corner), dot(m[1], corner), dot(m[2], corner));
                float2 q = to_light(s, world).xy;
                lo = min(lo, q);
                hi = max(hi, q);
            }
            Atomic<uint> *bound = p->bounds + 4 * (j * LEVELS + i);
            // Untouched, the level holds no page.
            if (bound[0].load() >= TILES)
                continue;
            int2 held_lo = int2(bound[0].load(), bound[1].load());
            int2 held_hi = int(TILES) - 1 - int2(bound[2].load(), bound[3].load());
            int2 a = max(tile_at(s, lo, i), held_lo);
            int2 b = min(tile_at(s, hi, i), held_hi);
            for (int y = a.y; y <= b.y; y++)
            {
                for (int x = a.x; x <= b.x; x++)
                {
                    uint page = s->tiles[(i * TILES + y) * TILES + x];
                    if (page == NO_PAGE)
                        continue;
                    uint d = p->state[0].increment();
                    if (d >= p->capacity)
                    {
                        p->state[4].increment();
                        continue;
                    }
                    p->draws[5 * d] = v->records[5 * r].load();
                    p->draws[5 * d + 1] = 1;
                    p->draws[5 * d + 2] = v->records[5 * r + 2].load();
                    p->draws[5 * d + 3] = v->records[5 * r + 3].load();
                    p->draws[5 * d + 4] = d;
                    p->pairs[d] = uint2(instance, page);
                }
            }
        }
    }
}

struct Varyings
{
    float4 position : SV_Position;
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
    uint2 owner = p->owners[pair.y];
    Shadow *s = p->shadows + owner.x;
    uint i = owner.y / TILEMAP;
    int2 t = int2(owner.y % TILES, owner.y / TILES % TILES) + s->origin[i];
    uint point = v->vertices[vertex].x;
    float4 local = float4(v->points[3 * point], v->points[3 * point + 1], v->points[3 * point + 2], 1);
    float4 *m = v->instances + 7 * pair.x;
    float3 world = float3(dot(m[0], local), dot(m[1], local), dot(m[2], local));
    float3 q = to_light(s, world);
    Varyings o;
    o.position = float4((q.xy / tile_width(s, i) - float2(t)) * 2 - 1, q.z, 1);
    o.page = pair.y;
    o.layer = pair.y;
    return o;
}

[shader("fragment")]
void fs(Varyings v)
{
    uint2 x = uint2(v.position.xy);
    // Depth is in [0, 1], where float bits order as integers do.
    push.pages->pool[(v.page * PAGE + x.y) * PAGE + x.x].min(asuint(v.position.z));
}
