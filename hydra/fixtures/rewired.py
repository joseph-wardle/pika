# Moving a connection changes the network's structure, not a value.
from pxr import Sdf, UsdShade
surface = UsdShade.Shader.Get(stage, "/world/mtl/paint/surface")
tex = UsdShade.Shader.Get(stage, "/world/mtl/paint/tex")
surface.GetInput("diffuseColor").DisconnectSource()
surface.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
