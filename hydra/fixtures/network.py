# A value edit arrives as the whole network again, and so does a new asset
# path, resolved: neither compiles a surface, the new file is baked.
from pxr import Sdf
stage.GetAttributeAtPath("/world/mtl/paint/surface.inputs:roughness").Set(0.75)
stage.GetAttributeAtPath("/world/mtl/paint/tex.inputs:file").Set(Sdf.AssetPath("../../bake/fixtures/mips.exr"))
