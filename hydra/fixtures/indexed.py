# Changes the indices only: the flattened values must follow.
from pxr import Vt
stage.GetAttributeAtPath("/world/tri.primvars:normals:indices").Set(Vt.IntArray([1, 1, 0]))
