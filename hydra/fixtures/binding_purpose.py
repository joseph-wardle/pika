# Without a full binding, Hydra drops the subset and gives the mesh no material: the preview
# binding must not stand in for either.
stage.GetPrimAtPath("/world/tri/all").RemoveProperty("material:binding:full")
stage.GetPrimAtPath("/world/whole").RemoveProperty("material:binding:full")
