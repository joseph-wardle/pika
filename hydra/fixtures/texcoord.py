# The name a reader reads is compiled into the surface, so this edit compiles
# one, and the mesh must carry the other primvar after it.
stage.GetAttributeAtPath("/world/mtl/paint/uv.inputs:varname").Set("st")
