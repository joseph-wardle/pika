# Moves the native instance inside the point instancer's prototype.
from pxr import Gf
stage.GetAttributeAtPath("/world/scatter/prototypes/leaf.xformOp:translate").Set(Gf.Vec3d(0, 5, 0))
