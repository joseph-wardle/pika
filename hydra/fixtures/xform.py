# Moves the parent only: the mesh's transform must follow.
from pxr import Gf
stage.GetAttributeAtPath("/world/arm.xformOp:translate").Set(Gf.Vec3d(5, 0, 0))
