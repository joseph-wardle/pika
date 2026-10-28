# Without its full binding, Hydra drops the subset (its faces take the mesh's
# material): the preview binding must not stand in for it.
stage.GetPrimAtPath("/world/tri/all").RemoveProperty("material:binding:full")
