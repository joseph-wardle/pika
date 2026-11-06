# A wrap mode is a value: the edit rewrites the material's block and rebinds
# its sampler, compiling and baking nothing.
stage.GetAttributeAtPath("/world/mtl/paint/tex.inputs:wrapS").Set("repeat")
