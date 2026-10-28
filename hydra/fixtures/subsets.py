# Swaps the faces, keeping the bindings.
from pxr import Vt
stage.GetAttributeAtPath("/world/quad/left.indices").Set(Vt.IntArray([1]))
stage.GetAttributeAtPath("/world/quad/right.indices").Set(Vt.IntArray([0]))
