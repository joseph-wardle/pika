# Changes the normals only, so this sync depends on DirtyNormals.
from pxr import Gf, Vt
stage.GetAttributeAtPath("/world/tri.normals").Set(Vt.Vec3fArray([Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 0, 1)]))
