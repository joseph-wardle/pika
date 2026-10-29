"""Pika's image check: renders each scene in hydra/images/ in husk, for
frames 1 and 2, and compares its color, depth and normal AOVs with a ray cast
from USD, one ray through each pixel centre. The reference reads the stage,
never Pika's store, so the two cannot share a mistake.

    hython hydra/image.py <out dir> [usdrecord]

Run it from a shell set up as hydra/check.sh sets one up. With `usdrecord`,
it also renders each scene's color through usdrecord (usdview's engine),
which needs an X display in DISPLAY.

What the reference encodes:
- The camera is the RenderSettings camera's frustum, conformed to the
  image's aspect (Fit, which is expandAperture). Pixel rows go down, as in
  the EXR. No host hands Pika an off-centre data window (see plugin.cpp),
  so every pixel is rendered.
- Meshes are drawn once per instance: native instances through their
  instance proxies, point instancers' prototypes at USD's instance
  transforms, without the instances its mask (invisibleIds) hides. Only
  meshes of the host's purposes are drawn: husk's default and render,
  usdrecord's default and proxy. Nothing is culled: Pika keeps an instance
  unless its box is wholly outside one plane of the view's frustum, so it
  must draw exactly what the reference does.
- A mesh whose subdivisionScheme is not none is drawn refined once, as
  Pika draws it (one level, whatever the host's refine level): each face
  splits into quads, one per corner, through the corner's vertex point, its
  edges' points and the face point, starting at the vertex point, except
  that a quad's child of corner j has its vertex point at j. Bilinear
  points are averages. Catmull-Clark points follow the original rules
  with USD's default boundary, edgeAndCorner: boundary edges are
  midpoints, boundary vertices (6v + a + b) / 8 of their boundary
  neighbours, and vertices of a single face stay put. Holes refine with
  the rest and their child faces are dropped. Other schemes, boundary
  rules, creases and corners are refused, not approximated.
- Faces but the holes split into fans around their first vertex, as Hydra
  splits them.
- Depth is OpenGL's window depth, (z_ndc + 1) / 2, of the hit point.
- The normal is the authored normal on a mesh whose subdivisionScheme is
  none. Otherwise, or without one, it is the point's smooth normal on the
  drawn (refined) faces, as Hydra computes it on its faces: the sum over the point's corners, holes included, of the
  cross product of the edges to the next and previous corner (swapped when
  left-handed), normalized. Either is interpolated, moved by the inverse
  transpose of the world matrix and normalized. A face's front is the side
  its moved geometric normal points to, reversed when left-handed; seen from
  behind, the normal is negated. Color is grey: the normal against the
  direction toward the eye, clamped, with alpha 1.
- Pixels with no hit keep the clear values: color 0, depth 1, normal 0.

Pixels next to a change of face (silhouettes, face boundaries) are compared
only for coverage, since a rasterizer and a ray decide them differently. Away
from them, values must agree within TOLERANCE.

Each control perturbs the reference. A control must fail on every scene
and host whose reference it changes, and must change at least one scene's.
"""

import math
import os
import subprocess
import sys

import numpy as np
import OpenImageIO as oiio
from pxr import CameraUtil, Gf, Usd, UsdGeom, UsdRender

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = os.path.join(ROOT, "hydra", "images")
SETTINGS = "/render/settings"
FRAMES = (1, 2)
# Color allows for usdrecord's half floats.
TOLERANCE = {"color": 2e-3, "depth": 2e-6, "normal": 2e-4}
CONTROLS = ["focal length +0.5 %", "half a pixel right", "aperture cropped", "unconverted depth",
            "unflipped normals", "flat normals", "instancer transform dropped", "outer instances dropped",
            "orientations misread", "authored normals on subdivided meshes", "holes drawn",
            "face normals averaged", "left-handed corners unswapped", "frame 1's smooth normals",
            "cage drawn on subdivided meshes", "boundary rules of the interior",
            "purposes ignored", "culled by box centre", "frame 1's reference"]
HUSK_PURPOSES = ("default", "render")
USDRECORD_PURPOSES = ("default", "proxy")


def meshes(stage, time, purposes, control=None):
    """Every mesh of `purposes` drawn, once per instance it is drawn at: its
    prim and world matrix. Native instances are walked through their instance
    proxies; a point instancer's prototypes are drawn only at its instances,
    by USD's ComputeInstanceTransformsAtTime, which includes each prototype
    root's own transform."""
    out = []
    walked = set()

    def walk(prim, parent, local=None):
        imageable = UsdGeom.Imageable(prim)
        if imageable and imageable.GetVisibilityAttr().Get(time) == UsdGeom.Tokens.invisible:
            return
        if local is None:
            xformable = UsdGeom.Xformable(prim)
            local = xformable.GetLocalTransformation(time) if xformable else Gf.Matrix4d(1)
        if control == "outer instances dropped" and prim.IsInstance():
            # Each prototype once, in its own space.
            if prim.GetPrototype() in walked:
                return
            walked.add(prim.GetPrototype())
            local = parent = Gf.Matrix4d(1)
        world = local * parent
        if prim.IsA(UsdGeom.PointInstancer):
            instancer = UsdGeom.PointInstancer(prim)
            prototypes = instancer.GetPrototypesRel().GetTargets()
            indices = instancer.GetProtoIndicesAttr().Get(time)
            transforms = instancer.ComputeInstanceTransformsAtTime(
                time, time, UsdGeom.PointInstancer.IncludeProtoXform, UsdGeom.PointInstancer.IgnoreMask)
            mask = instancer.ComputeMaskAtTime(time)
            if control == "instancer transform dropped":
                world = parent
            for i, transform in enumerate(transforms):
                if not mask or mask[i]:
                    walk(stage.GetPrimAtPath(prototypes[indices[i]]), world, transform)
            return
        drawn = imageable and imageable.ComputePurpose() in purposes
        if prim.IsA(UsdGeom.Mesh) and (drawn or control == "purposes ignored"):
            out.append((prim, world))
        for child in prim.GetFilteredChildren(Usd.TraverseInstanceProxies()):
            walk(child, world)

    walk(stage.GetPseudoRoot(), Gf.Matrix4d(1))
    return out


def misread_orientations(stage, time):
    """Authors, in the session layer, each point instancer's orientations
    as they would read if GfQuatf's (i, j, k, real) were taken for
    (real, i, j, k)."""
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in stage.TraverseAll():
            attr = UsdGeom.PointInstancer(prim).GetOrientationsAttr() if prim.IsA(UsdGeom.PointInstancer) else None
            if attr and attr.HasAuthoredValue():
                attr.Set([Gf.Quath(q.GetImaginary()[0], Gf.Vec3h(q.GetImaginary()[1], q.GetImaginary()[2],
                                                                 q.GetReal())) for q in attr.Get(time)])


def smooth_normals(points, counts, indices, left, control=None):
    """Each point's smooth normal, as Hydra's Hd_SmoothNormals computes it."""
    sums = [Gf.Vec3d(0) for _ in points]
    corner = 0
    for count in counts:
        face = [indices[corner + j] for j in range(count)]
        corner += count
        p = [Gf.Vec3d(points[i]) for i in face]
        unit = Gf.Cross(p[1] - p[0], p[2] - p[0]).GetNormalized() if count > 2 else Gf.Vec3d(0)
        for j, curr in enumerate(face):
            prev, next = p[j - 1], p[(j + 1) % count]
            if left and control != "left-handed corners unswapped":
                prev, next = next, prev
            if control == "face normals averaged":
                sums[curr] += -unit if left else unit
            else:
                sums[curr] += Gf.Cross(next - p[j], prev - p[j])
    return [n.GetNormalized() if n.GetLength() > 0 else n for n in sums]


def refine(points, counts, indices, holes, scheme, control=None):
    """One level of uniform refinement: the refined points, face vertex
    counts, face vertex indices and holes."""
    points = [Gf.Vec3d(p) for p in points]
    faces, corner = [], 0
    for count in counts:
        faces.append(indices[corner:corner + count])
        corner += count
    edges = {}
    for f, face in enumerate(faces):
        for j, v in enumerate(face):
            edges.setdefault(frozenset((v, face[(j + 1) % len(face)])), []).append(f)
    if any(len(fs) > 2 for fs in edges.values()):
        raise ValueError("the reference does not refine non-manifold edges")
    centre = [sum((points[v] for v in face), Gf.Vec3d(0)) / len(face) for face in faces]
    edge_point = {}
    for e, fs in edges.items():
        a, b = (points[v] for v in e)
        boundary = len(fs) == 1 and control != "boundary rules of the interior"
        if scheme == UsdGeom.Tokens.bilinear or boundary:
            edge_point[e] = (a + b) / 2
        else:
            edge_point[e] = (a + b + sum((centre[f] for f in fs), Gf.Vec3d(0))) / (2 + len(fs))
    vertex_point = list(points)
    if scheme == UsdGeom.Tokens.catmullClark:
        for v, p in enumerate(points):
            incident = [f for f, face in enumerate(faces) if v in face]
            around = [e for e in edges if v in e]
            boundary = [e for e in around if len(edges[e]) == 1]
            if control == "boundary rules of the interior":
                boundary = []
            if len(incident) == 1 and boundary:
                continue
            if boundary:
                a, b = (points[next(iter(e - {v}))] for e in boundary)
                vertex_point[v] = (6 * p + a + b) / 8
                continue
            n = len(around)
            q = sum((centre[f] for f in incident), Gf.Vec3d(0)) / len(incident)
            r = sum((sum((points[u] for u in e), Gf.Vec3d(0)) / 2 for e in around), Gf.Vec3d(0)) / n
            vertex_point[v] = (q + 2 * r + (n - 3) * p) / n
    refined = vertex_point
    index = {}
    for e, point in edge_point.items():
        index[e] = len(refined)
        refined.append(point)
    out_counts, out_indices, out_holes = [], [], []
    for f, face in enumerate(faces):
        middle = len(refined)
        refined.append(centre[f])
        for j, v in enumerate(face):
            after = index[frozenset((v, face[(j + 1) % len(face)]))]
            before = index[frozenset((face[j - 1], v))]
            if f in holes:
                out_holes.append(len(out_counts))
            out_counts.append(4)
            child = [v, after, middle, before]
            # OpenSubdiv rotates a quad's children to keep its parametric
            # orientation, which decides the diagonal each child is split on.
            rotation = j if len(face) == 4 else 0
            out_indices += child[-rotation:] + child[:-rotation] if rotation else child
    return refined, out_counts, out_indices, set(out_holes)


def surface(mesh, time, control=None):
    """The points, face vertex counts, face vertex indices and holes a
    mesh is drawn with at `time`."""
    points = mesh.GetPointsAttr().Get(time)
    counts = mesh.GetFaceVertexCountsAttr().Get(time)
    indices = mesh.GetFaceVertexIndicesAttr().Get(time)
    holes = set(mesh.GetHoleIndicesAttr().Get(time) or [])
    scheme = mesh.GetSubdivisionSchemeAttr().Get(time)
    if scheme == UsdGeom.Tokens.none or control in ("cage drawn on subdivided meshes",
                                                    "authored normals on subdivided meshes"):
        return points, counts, indices, holes
    if scheme not in (UsdGeom.Tokens.catmullClark, UsdGeom.Tokens.bilinear):
        raise ValueError(f"{mesh.GetPath()}: the reference does not refine {scheme}")
    authored = [mesh.GetCreaseIndicesAttr(), mesh.GetCornerIndicesAttr()]
    if any(a.HasAuthoredValue() for a in authored) or (
            mesh.GetInterpolateBoundaryAttr().Get(time) != UsdGeom.Tokens.edgeAndCorner):
        raise ValueError(f"{mesh.GetPath()}: the reference refines only uncreased edgeAndCorner meshes")
    return refine(points, counts, indices, holes, scheme, control)


def triangles(stage, time, purposes, control=None):
    """Every drawn mesh instance's fan triangles, in world space: for each,
    the world points, the object normals at its corners, the normal matrix,
    whether it is left-handed, and a face id."""
    out = []
    for instance, (prim, world) in enumerate(meshes(stage, time, purposes, control)):
        mesh = UsdGeom.Mesh(prim)
        # Gf multiplies row vectors, so normals go through the inverse transpose.
        normal_matrix = world.ExtractRotationMatrix().GetInverse().GetTranspose()
        points, counts, indices, holes = surface(mesh, time, control)
        cage = Gf.Range3d()
        for p in mesh.GetPointsAttr().Get(time):
            cage.UnionWith(Gf.Vec3d(p))
        centre = world.Transform(cage.GetMidpoint())
        if control == "holes drawn":
            holes = set()
        normals = mesh.GetNormalsAttr().Get(time)
        interpolation = mesh.GetNormalsInterpolation()
        left = mesh.GetOrientationAttr().Get(time) == UsdGeom.Tokens.leftHanded
        subdivided = mesh.GetSubdivisionSchemeAttr().Get(time) != UsdGeom.Tokens.none
        if control == "authored normals on subdivided meshes":
            subdivided = False
        if subdivided or not normals:
            # The points smooth normals are computed from.
            at = FRAMES[0] if control == "frame 1's smooth normals" else time
            normals = smooth_normals(surface(mesh, at, control)[0], counts, indices, left, control)
            interpolation = UsdGeom.Tokens.vertex
        corner = 0
        for face, count in enumerate(counts):
            if face in holes:
                corner += count
                continue

            def normal(c):
                if interpolation == UsdGeom.Tokens.uniform:
                    return Gf.Vec3d(normals[face])
                if interpolation == UsdGeom.Tokens.faceVarying:
                    return Gf.Vec3d(normals[corner + c])
                return Gf.Vec3d(normals[indices[corner + c]])

            for k in range(1, count - 1):
                cs = (0, k, k + 1)
                out.append({
                    "points": [world.Transform(Gf.Vec3d(points[indices[corner + c]])) for c in cs],
                    "object": [Gf.Vec3d(points[indices[corner + c]]) for c in cs],
                    "normals": [normal(c) for c in cs],
                    "normal_matrix": normal_matrix,
                    "left": left,
                    "centre": centre,
                    "face": (str(prim.GetPath()), instance, face),
                })
            corner += count
    return out


def frustum(stage, time, width, height, focal, policy):
    settings = UsdRender.Settings(stage.GetPrimAtPath(SETTINGS))
    camera_path = settings.GetCameraRel().GetTargets()[0]
    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path)).GetCamera(time)
    camera.focalLength *= focal
    f = camera.frustum
    CameraUtil.ConformWindow(f, policy, width / height)
    return f


def reference(stage, time, width, height, purposes, control=None):
    focal = 1.005 if control == "focal length +0.5 %" else 1.0
    policy = CameraUtil.Crop if control == "aperture cropped" else CameraUtil.Fit
    f = frustum(stage, time, width, height, focal, policy)
    to_clip = f.ComputeViewMatrix() * f.ComputeProjectionMatrix()
    eye = f.position
    if control == "orientations misread":
        misread_orientations(stage, time)
    tris = triangles(stage, time, purposes, control)
    stage.GetSessionLayer().Clear()
    if control == "culled by box centre":
        def seen(p):
            c = to_clip.Transform(p)
            return f.GetNearFar().Contains(-f.ComputeViewMatrix().Transform(p)[2]) and all(abs(v) <= 1 for v in (c[0], c[1]))
        tris = [t for t in tris if seen(t["centre"])]
    shift = 0.5 if control == "half a pixel right" else 0.0

    image = {
        "color": np.zeros((height, width, 4)),
        "depth": np.ones((height, width)),
        "normal": np.zeros((height, width, 3)),
        "face": np.full((height, width), -1),
    }
    faces = {}
    for y in range(height):
        for x in range(width):
            ndc = Gf.Vec2d(2 * (x + 0.5 + shift) / width - 1, 1 - 2 * (y + 0.5) / height)
            ray = f.ComputePickRay(ndc)
            best = None
            for t in tris:
                hit, distance, bary, _ = ray.Intersect(*t["points"])
                if hit and (best is None or distance < best[0]):
                    best = (distance, bary, t)
            if best is None:
                continue
            distance, bary, t = best
            p = ray.GetPoint(distance)
            m = t["normal_matrix"]
            o = t["object"]
            front = Gf.Cross(o[1] - o[0], o[2] - o[0]) * m
            if t["left"]:
                front = -front
            toward = (eye - p).GetNormalized()
            if control == "flat normals":
                n = front.GetNormalized()
            else:
                n = sum((b * c for b, c in zip(bary, t["normals"])), Gf.Vec3d(0)) * m
                n = n.GetNormalized()
            if Gf.Dot(front, toward) < 0 and control != "unflipped normals":
                n = -n
            if control == "flat normals" and Gf.Dot(n, toward) < 0:
                n = -n
            grey = min(max(Gf.Dot(n, toward), 0.0), 1.0)
            depth = (to_clip.Transform(p)[2] + 1) / 2
            if control == "unconverted depth":
                depth = 1 - depth
            image["color"][y, x] = (grey, grey, grey, 1)
            image["depth"][y, x] = depth
            image["normal"][y, x] = tuple(n)
            image["face"][y, x] = faces.setdefault(t["face"], len(faces))
    return image


def read(path):
    """An EXR's AOVs, by subimage name."""
    image = {}
    for i in range(oiio.ImageBuf(path).nsubimages):
        buf = oiio.ImageBuf(path, i, 0)
        pixels = buf.get_pixels(oiio.FLOAT)
        # husk names the color subimage C, usdrecord beauty.
        name = buf.spec().get_string_attribute("oiio:subimagename") or "color"
        image[{"C": "color", "beauty": "color"}.get(name, name)] = pixels[..., 0] if buf.nchannels == 1 else pixels
    return image


def edges(face):
    """Pixels whose 3×3 neighbourhood holds more than one face (or none)."""
    padded = np.pad(face, 1, mode="edge")
    h, w = face.shape
    out = np.zeros(face.shape, bool)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy:dy + h, dx:dx + w] != face
    return out


def compare(image, ref, aovs):
    """Why `image` does not match `ref`, or an empty list."""
    problems = []
    covered = ref["face"] >= 0
    edge = edges(ref["face"])
    drawn = image["color"][..., 3] > 0.5
    stray = (drawn != covered) & ~edge
    if stray.any():
        problems.append(f"coverage differs at {stray.sum()} pixels away from edges")
    inner = covered & drawn & ~edge
    for aov in aovs:
        error = np.abs(image[aov][..., :4] - ref[aov][..., :4] if aov == "color" else image[aov] - ref[aov])
        worst = error[inner].max() if inner.any() else 0.0
        if worst > TOLERANCE[aov]:
            problems.append(f"{aov} differs by up to {worst:.3g} (tolerance {TOLERANCE[aov]:g})")
    return problems


def husk(scene, out):
    name = os.path.splitext(os.path.basename(scene))[0]
    log = os.path.join(out, f"{name}.husk.log")
    with open(log, "w") as f:
        subprocess.run(["husk", "-V", "1", "-R", "Pika", "-s", SETTINGS, "--disable-motionblur",
                        "-f", str(FRAMES[0]), "-n", str(len(FRAMES)),
                        "-o", os.path.join(out, f"{name}.husk.$F4.exr"), scene],
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    return {t: read(os.path.join(out, f"{name}.husk.{t:04d}.exr")) for t in FRAMES}, log


def usdrecord(scene, out, width):
    name = os.path.splitext(os.path.basename(scene))[0]
    log = os.path.join(out, f"{name}.usdrecord.log")
    with open(log, "w") as f:
        subprocess.run(["usdrecord", "--renderer", "Pika", "--camera", "/camera",
                        "--imageWidth", str(width), "--colorCorrectionMode", "disabled",
                        "--frames", f"{FRAMES[0]}:{FRAMES[-1]}",
                        scene, os.path.join(out, f"{name}.usdrecord.#.exr")],
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    return {t: read(os.path.join(out, f"{name}.usdrecord.{t}.exr")) for t in FRAMES}, log


def references(stage, width, height, purposes):
    """The reference and each control's, for each frame."""
    controls = [None] + [c for c in CONTROLS if c != "frame 1's reference"]
    return {t: {c: reference(stage, t, width, height, purposes, c) for c in controls} for t in FRAMES}


def check(name, renders, references, aovs):
    """Compares a host's frames with the references and the controls.
    Returns (failures, the controls that changed a reference)."""
    failures = []
    changed = set()
    for t in FRAMES:
        problems = compare(renders[t], references[t][None], aovs)
        print(f"  frame {t}: {'; '.join(problems) or 'matches'}")
        failures += [f"{name} frame {t}: {p}" for p in problems]
    t = FRAMES[-1]
    for control in CONTROLS:
        frame = FRAMES[0] if control == "frame 1's reference" else t
        perturbed = references[frame][None if frame != t else control]
        if not compare(perturbed, references[t][None], aovs):
            continue
        changed.add(control)
        caught = bool(compare(renders[t], perturbed, aovs))
        print(f"  control '{control}': {'fails, as it must' if caught else 'PASSES'}")
        if not caught:
            failures.append(f"{name}: the control '{control}' passes")
    return failures, changed


def main():
    out = sys.argv[1]
    record = sys.argv[2:] == ["usdrecord"]
    os.makedirs(out, exist_ok=True)
    scenes = sorted(os.path.join(SCENES, s) for s in os.listdir(SCENES) if s != "settings.usda")
    failures = []
    changed = set()
    for scene in scenes:
        name = os.path.splitext(os.path.basename(scene))[0]
        stage = Usd.Stage.Open(scene)
        width, height = UsdRender.Settings(stage.GetPrimAtPath(SETTINGS)).GetResolutionAttr().Get()
        refs = references(stage, width, height, HUSK_PURPOSES)
        renders, log = husk(scene, out)
        print(f"{name} in husk ({log}):")
        f, c = check(name, renders, refs, ["color", "depth", "normal"])
        failures += f
        changed |= c
        if not record:
            continue
        # usdrecord takes the image's height from the camera's aspect.
        renders, log = usdrecord(scene, out, width)
        shape = renders[FRAMES[0]]["color"].shape
        purposed = any(UsdGeom.Imageable(p).GetPurposeAttr().HasAuthoredValue() for p in stage.Traverse())
        if shape[:2] != (height, width) or purposed:
            refs = references(stage, shape[1], shape[0], USDRECORD_PURPOSES)
        print(f"{name} in usdrecord ({log}):")
        failures += check(f"{name} (usdrecord)", renders, refs, ["color"])[0]
    for control in CONTROLS:
        if control not in changed:
            failures.append(f"the control '{control}' changes no scene's reference")
    for f in failures:
        print(f"FAIL: {f}")
    print("PASS" if not failures else f"{len(failures)} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
