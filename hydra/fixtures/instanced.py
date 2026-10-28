# Moves one instance: only its matrix may change.
from pxr import Gf
stage.GetAttributeAtPath("/world/b.xformOp:translate").Set(Gf.Vec3d(0, 3, 0))
