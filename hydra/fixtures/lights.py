# Edits the sun's inputs and hides it, turns its parent, and removes the moon.
from pxr import UsdGeom
stage.GetAttributeAtPath("/world/rig/sun.inputs:intensity").Set(2.0)
stage.GetAttributeAtPath("/world/rig/sun.inputs:normalize").Set(True)
UsdGeom.Imageable(stage.GetPrimAtPath("/world/rig/sun")).MakeInvisible()
stage.GetAttributeAtPath("/world/rig.xformOp:rotateY").Set(90)
stage.RemovePrim("/world/moon")
