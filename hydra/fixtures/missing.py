# Names a file that resolves: the material is tried again, bakes it and compiles.
from pxr import Sdf
stage.GetAttributeAtPath("/world/mtl/skin/tex.inputs:file").Set(Sdf.AssetPath("../../bake/fixtures/swatch.png"))
