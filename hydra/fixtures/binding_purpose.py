# Without a full binding, Hydra drops a subset that inherits none, and gives
# the mesh no material: the preview binding must not stand in for either.
stage.GetPrimAtPath("/world/tri/all").RemoveProperty("material:binding:full")
stage.GetPrimAtPath("/world/whole").RemoveProperty("material:binding:full")
