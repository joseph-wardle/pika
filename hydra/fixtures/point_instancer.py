# Moves one point: only the positions primvar may change.
from pxr import Gf, Vt
stage.GetAttributeAtPath("/world/scatter.positions").Set(
    Vt.Vec3fArray([Gf.Vec3f(1, 0, 0), Gf.Vec3f(3, 0, 0)]))
