from pxr import Sdf, UsdGeom
stage.GetAttributeAtPath("/world/rig/sun.inputs:intensity").Set(2.0)
stage.GetAttributeAtPath("/world/rig/sun.inputs:normalize").Set(True)
UsdGeom.Imageable(stage.GetPrimAtPath("/world/rig/sun")).MakeInvisible()
stage.GetAttributeAtPath("/world/rig.xformOp:rotateY").Set(90)
stage.RemovePrim("/world/moon")
stage.GetPrimAtPath("/world/bulb").CreateAttribute("inputs:radius", Sdf.ValueTypeNames.Float).Set(2.0)
stage.GetPrimAtPath("/world/tube").CreateAttribute("inputs:length", Sdf.ValueTypeNames.Float).Set(4.0)
stage.GetAttributeAtPath("/world/panel.inputs:shadow:enable").Set(True)
