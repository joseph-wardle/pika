"""The ALab oracle: derives Hydra's view of a sample of a stage's meshes from USD and checks
Pika's store against it. Trust it only while hydra/check.sh's fixtures pass.

    hython hydra/oracle.py <stage.usda> <frame> <out dir>

Run it in hydra/check.sh's environment. The sample is visible, renderable, non-instanced meshes,
half static and half animated, and a layer deactivates the rest. Instance prototypes are left to
the fixtures, as Hydra renames them.

Controls: each mesh's store against the next mesh's expectation, and each animated mesh's against
the next frame's.
"""

import os
import re
import subprocess
import sys

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdShade

SAMPLE = 16


def candidates(stage, time):
    """Not instanced, visible, of husk's default purposes, and not replaced by a draw mode."""
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility(time) != UsdGeom.Tokens.inherited:
            continue
        if imageable.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue
        ancestors = [prim.GetParent()]
        while not ancestors[-1].IsPseudoRoot():
            ancestors.append(ancestors[-1].GetParent())
        if any(a.IsA(UsdGeom.PointInstancer) for a in ancestors):
            continue
        if any(UsdGeom.ModelAPI(a).GetModelDrawModeAttr().HasAuthoredValue() for a in ancestors):
            continue
        yield prim


def spread(prims, n):
    prims = sorted(prims, key=lambda p: p.GetPath())
    if len(prims) <= n:
        return prims
    return [prims[i * len(prims) // n] for i in range(n)]


def expected(prim, time):
    mesh = UsdGeom.Mesh(prim)
    purpose = UsdGeom.Imageable(prim).ComputePurpose()
    matrix = mesh.ComputeLocalToWorldTransform(time)
    bound = lambda p: UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial(UsdShade.Tokens.full)[0]
    material = bound(prim)
    subsets = []
    for subset in UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets():
        # A subset inherits the mesh's binding. Hydra drops one with no
        # material for the delegate's purpose.
        faces = bound(subset.GetPrim())
        if faces:
            subsets.append((str(faces.GetPath()), list(subset.GetIndicesAttr().Get(time))))
    primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar(UsdGeom.Tokens.normals)
    if primvar and primvar.HasValue():
        normals = (primvar.GetInterpolation(), primvar.ComputeFlattened(time))
    elif mesh.GetNormalsAttr().HasValue():
        normals = (mesh.GetNormalsInterpolation(), mesh.GetNormalsAttr().Get(time))
    else:
        normals = None
    return {
        "render_tag": "geometry" if purpose == UsdGeom.Tokens.default_ else str(purpose),
        "visible": "true",
        "instancer": "none",
        "transform": np.array([matrix[i][j] for i in range(4) for j in range(4)]),
        "material": str(material.GetPath()) if material else "",
        "orientation": str(mesh.GetOrientationAttr().Get(time)),
        "face_vertex_counts": list(mesh.GetFaceVertexCountsAttr().Get(time)),
        "face_vertex_indices": list(mesh.GetFaceVertexIndicesAttr().Get(time)),
        "subsets": subsets,
        "points": np.array(mesh.GetPointsAttr().Get(time), dtype=np.float32).reshape(-1),
        "normals": None if normals is None else
            (str(normals[0]), np.array(normals[1], dtype=np.float32).reshape(-1)),
    }


def moves(prim, time):
    mesh = UsdGeom.Mesh(prim)
    after = Usd.TimeCode(time.GetValue() + 1)
    if mesh.ComputeLocalToWorldTransform(time) != mesh.ComputeLocalToWorldTransform(after):
        return True
    points = mesh.GetPointsAttr()
    return points.ValueMightBeTimeVarying() and points.Get(time) != points.Get(after)


def over(stage):
    """Keeps the root's timing and units."""
    root = stage.GetRootLayer()
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.subLayerPaths.append(root.realPath)
    # A sublayer's time is scaled by its timeCodesPerSecond against ours.
    for key in ("timeCodesPerSecond", "framesPerSecond", "startTimeCode", "endTimeCode",
                "metersPerUnit", "upAxis"):
        if root.pseudoRoot.HasInfo(key):
            layer.pseudoRoot.SetInfo(key, root.pseudoRoot.GetInfo(key))
    return layer


def write_sample_layer(stage, keep, path):
    layer = over(stage)
    ancestors = {a for p in keep for a in p.GetPrefixes()}
    for parent in sorted(ancestors | {Sdf.Path.absoluteRootPath}):
        if parent in keep:
            continue
        for child in stage.GetPrimAtPath(parent).GetChildren():
            if child.GetPath() not in ancestors:
                Sdf.CreatePrimInLayer(layer, child.GetPath()).active = False
    layer.Export(path)


def parse_dump(path):
    text = open(path).read()
    last = text.rsplit("\nsync ", 1)[-1] if "\nsync " in text else text
    meshes, mesh, field = {}, None, None
    for line in last.splitlines()[1:]:
        if line.startswith("mesh "):
            mesh = meshes.setdefault(line[5:], {"subsets": [], "rows": {}})
        elif not line.startswith(" "):
            mesh = None
        elif mesh is None:
            continue
        elif not line.startswith("   "):
            field, _, rest = line.strip().partition(" ")
            mesh[field] = rest
            mesh["rows"][field] = []
        elif field == "topology":
            line = line.strip()
            name, _, rest = line.partition(" ")
            if name == "subset":
                material, faces = re.fullmatch(r'subset "(.*)" faces(.*)', line).groups()
                mesh["subsets"].append((material, [int(f) for f in faces.split()]))
            else:
                mesh[name] = [int(v) for v in rest.split()]
        else:
            mesh["rows"][field].append(line)
    for mesh in meshes.values():
        rows = mesh.pop("rows")
        floats = lambda f: np.array(" ".join(rows.get(f, [])).split(), dtype=np.float64)
        mesh["transform"] = floats("transform")
        mesh["orientation"] = mesh.pop("topology", "not synced")
        mesh["material"] = mesh.get("material", "not synced").strip('"')
        mesh["points"] = floats("points").astype(np.float32)
        interpolation = mesh["normals"].split()[0]
        mesh["normals"] = None if interpolation == "none" else \
            (interpolation, floats("normals").astype(np.float32))
    return meshes


def differences(actual, want):
    def same(a, b):
        if isinstance(b, np.ndarray):
            return isinstance(a, np.ndarray) and np.array_equal(a, b)
        if isinstance(b, tuple):
            return isinstance(a, tuple) and len(a) == len(b) and all(map(same, a, b))
        return a == b
    return [k for k in want if not same(actual.get(k), want[k])]


def main(stage_path, frame, out):
    os.makedirs(out, exist_ok=True)
    time = Usd.TimeCode(float(frame))
    stage = Usd.Stage.Open(stage_path)
    meshes = list(candidates(stage, time))
    moving = {p for p in meshes if moves(p, time)}
    animated = spread(moving, SAMPLE // 2)
    sample = spread(set(meshes) - moving, SAMPLE // 2) + animated
    print(f"oracle: {len(meshes)} candidate meshes, {len(moving)} animated; "
          f"sampling {len(sample)}")

    want = {str(p.GetPath()): expected(p, time) for p in sample}
    camera = min(str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Camera))
    keep = {p.GetPath() for p in sample} | {Sdf.Path(camera)}
    keep |= {Sdf.Path(m) for w in want.values() for m, _ in w["subsets"]}
    keep |= {Sdf.Path(w["material"]) for w in want.values() if w["material"]}
    layer = os.path.join(out, "sample.usda")
    write_sample_layer(stage, keep, layer)

    dump = os.path.join(out, "sample.dump")
    log = os.path.join(out, "sample.log")
    with open(log, "w") as f:
        status = subprocess.call(
            ["husk", "-V", "1", "-R", "Pika", "--headlight", "none", "-f", frame, "-n", "1",
             "--res", "64", "64", "-c", camera, "-o", os.path.join(out, "sample.exr"), layer],
            stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, PIKA_STORE_DUMP=dump))
    failed = []
    if status != 0:
        failed.append(f"husk exited {status} (see {log})")
    if any(line.startswith("pika: error") for line in open(log)):
        failed.append(f"Pika reported errors (see {log})")
    actual = parse_dump(dump)

    matched = 0
    for path, w in want.items():
        if path not in actual:
            failed.append(f"{path}: not in Pika's store")
            continue
        wrong = differences(actual[path], w)
        if wrong:
            failed.append(f"{path}: {', '.join(wrong)} differ")
        else:
            matched += 1
    print(f"oracle: {matched} of {len(want)} sampled meshes match at frame {frame}")
    if len(actual) != len(want):
        print(f"oracle: the store holds {len(actual)} meshes")

    paths = [p for p in want if p in actual]
    shifted = sum(1 for a, b in zip(paths, paths[1:] + paths[:1])
                  if not differences(actual[a], want[b]))
    after = Usd.TimeCode(time.GetValue() + 1)
    stale = sum(1 for p in animated if str(p.GetPath()) in actual and
                not differences(actual[str(p.GetPath())], expected(p, after)))
    print(f"oracle controls: {shifted} meshes match the next mesh's expectation, "
          f"{stale} animated meshes match the next frame's (both must be 0)")
    if len(paths) < 2 or shifted:
        failed.append("the shifted control did not fail")
    if not animated or stale:
        failed.append("the next-frame control did not fail")

    for message in failed:
        print("FAIL:", message)
    print("PASS" if not failed else "")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
