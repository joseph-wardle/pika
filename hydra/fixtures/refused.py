# Splits the quad into two triangles, which Loop refines.
stage.GetAttributeAtPath("/world/quad.faceVertexCounts").Set([3, 3])
stage.GetAttributeAtPath("/world/quad.faceVertexIndices").Set([0, 1, 2, 0, 2, 3])
