# Disconnects the texture: the material reads a colour, fits and builds.
from pxr import Gf
attr = stage.GetAttributeAtPath("/world/mtl/skin/surface.inputs:diffuseColor")
attr.ClearConnections()
attr.Set(Gf.Vec3f(0.2, 0.4, 0.6))
