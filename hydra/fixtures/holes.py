# Moves the hole and changes the scheme: both are topology.
stage.GetAttributeAtPath("/world/strip.holeIndices").Set([0])
stage.GetAttributeAtPath("/world/strip.subdivisionScheme").Set("bilinear")
