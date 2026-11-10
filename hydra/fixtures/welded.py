# Closing the seam welds the corners to fewer values, so the texcoord is refined afresh.
stage.GetAttributeAtPath("/world/strip.primvars:st").Set(
    [(0, 0), (0.3, 0), (0.3, 1), (0, 1), (0.3, 0), (0.6, 0), (0.6, 1), (0.3, 1),
     (0.6, 0), (1, 0), (1, 1), (0.6, 1)])
