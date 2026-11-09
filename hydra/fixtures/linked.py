# Links the key to both triangles and the rim to everything.
stage.GetRelationshipAtPath("/world/key.collection:lightLink:includes").AddTarget("/world/right")
stage.GetAttributeAtPath("/world/rim.collection:lightLink:includeRoot").Set(True)
