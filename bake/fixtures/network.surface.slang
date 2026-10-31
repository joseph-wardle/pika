// Written by src/surface.rs: the vertex stage for the pixel stage beside it.

#include "pixel.slang"
#include "vertex.slang"

[shader("vertex")]
VertexData vs(uint vertex : SV_VulkanVertexID, uint slot : SV_VulkanInstanceID)
{
    Vertex p = pull(vertex, slot);
    VertexData o;
    o.SV_Position = p.position;
    o.i_geomprop_st = p.texcoord;
    o.normalWorld = p.normal;
    o.tangentWorld = tangent(p.normal);
    o.bitangentWorld = cross(p.normal, tangent(p.normal));
    o.positionWorld = p.world;
    return o;
}
