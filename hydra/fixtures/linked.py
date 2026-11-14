stage.GetRelationshipAtPath("/world/key.collection:lightLink:includes").AddTarget("/world/right")
stage.GetAttributeAtPath("/world/rim.collection:lightLink:includeRoot").Set(True)
stage.GetRelationshipAtPath("/world/fill.collection:shadowLink:includes").SetTargets(["/world/right"])
