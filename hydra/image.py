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
  behind, the normal is negated.
- Color is the material's, with alpha 1. A mesh bound to a MaterialX
  surface whose emission is a uniform EDF of an image sampled at a primvar
  (hydra/images/textured.usda) shows the image's bilinear sample, level 0,
  at the hit's primvar, under the image node's address modes, with st
  (0, 0) at the image's bottom-left corner as USD and MaterialX read it, the
  channels padded with 0 to three. A mesh bound to a material whose surface
  shader names a MaterialX nodedef (ND_...) the host's libraries do not
  define is pink, (1, 0, 1), as Pika draws a material it cannot build,
  when the host hands the network over: husk and UsdImaging's scene-index
  path do, UsdImaging's legacy path drops a network whose surface shader
  the shader registry does not know, so Pika sees no surface and draws
  clay. usdrecord takes the path USDIMAGINGGL_ENGINE_ENABLE_SCENE_INDEX
  names, off by default in Houdini 21 and on in 22. Any other mesh is
  clay: grey, the normal against the direction toward the eye, clamped.
  Other materials are refused, not approximated.
- Pixels with no hit keep the clear values: color 0, depth 1, normal 0.

Pixels next to a change of face (silhouettes, face boundaries) are compared
only for coverage, since a rasterizer and a ray decide them differently. Away
from them, values must agree within TOLERANCE.

Each control perturbs the reference. A control must fail on every scene
and host whose reference it changes, and must change at least one scene's.
"""

import collections
import math
import os
import subprocess
import sys

import MaterialX as mx
import numpy as np
import OpenImageIO as oiio
from pxr import CameraUtil, Gf, Tf, Usd, UsdGeom, UsdImagingGL, UsdRender, UsdShade

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
            "purposes ignored", "culled by box centre", "headlight from the eye",
            "st origin at the top", "st wrapped, not clamped", "nearest texel",
            "failed materials drawn clay", "frame 1's reference"]
PINK = "pink"
# What a host hands Pika: the purposes it draws, and whether a material
# whose surface shader the shader registry does not know arrives with its
# network (see the module's doc).
Host = collections.namedtuple("Host", "purposes delivers_unknown")
HUSK = Host(("default", "render"), True)


def usdrecord_host():
    scene_index = Tf.GetEnvSetting("USDIMAGINGGL_ENGINE_ENABLE_SCENE_INDEX")
    return Host(("default", "proxy"), scene_index)


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


def upstream(shader, name):
    """The shader connected to `shader`'s input `name`, or None."""
    source = shader.GetInput(name).GetConnectedSource() if shader.GetInput(name) else None
    return UsdShade.Shader(source[0].GetPrim()) if source else None


def nodedefs():
    """The host's MaterialX libraries, as the bake tool loads them."""
    libraries = mx.createDocument()
    search = mx.FileSearchPath(os.path.join(os.environ["HFS"], "houdini", "materialx"))
    mx.loadLibraries(["libraries"], search, libraries)
    return libraries


NODEDEFS = nodedefs()


def unknown(shader):
    """Whether the shader names a nodedef the libraries do not define."""
    return shader.GetShaderId().startswith("ND_") and not NODEDEFS.getNodeDef(shader.GetShaderId())


def emission(prim, host, control=None):
    """The closed form of the mesh's material, or None for clay: PINK for a
    surface shader naming a nodedef the libraries do not define, when
    `host` delivers it; else the image file, its address modes (u, v) and
    the primvar it is sampled at, for a MaterialX surface whose emission is
    a uniform EDF of an image node's vector2 sample, through converts
    (hydra/images/textured.usda)."""
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not material:
        return None
    source = material.GetSurfaceOutput().GetConnectedSource()
    surface = UsdShade.Shader(source[0].GetPrim()) if source else None
    refuse = ValueError(f"{prim.GetPath()}: the reference knows only clay, pink and the emissive ramp")
    if surface and unknown(surface):
        clay = control == "failed materials drawn clay" or not host.delivers_unknown
        return None if clay else PINK
    if not surface or surface.GetShaderId() != "ND_surface":
        raise refuse
    edf = upstream(surface, "edf")
    if not edf or edf.GetShaderId() != "ND_uniform_edf" or upstream(surface, "bsdf"):
        raise refuse
    image = upstream(edf, "color")
    while image and image.GetShaderId().startswith("ND_convert_"):
        image = upstream(image, "in")
    if not image or image.GetShaderId() != "ND_image_vector2":
        raise refuse
    reader = upstream(image, "texcoord")
    if not reader or reader.GetShaderId() != "ND_geompropvalue_vector2":
        raise refuse
    modes = tuple(image.GetInput(m).Get() if image.GetInput(m) else "periodic"
                  for m in ("uaddressmode", "vaddressmode"))
    return image.GetInput("file").Get().resolvedPath, modes, reader.GetInput("geomprop").Get()


def sample(path, st, modes, control=None):
    """The bilinear sample of level 0 of the image at `path` at each st (an
    (n, 2) array), rows from the bottom as USD reads st, under the address
    modes (u, v); the channels padded with 0 to three."""
    image = oiio.ImageBuf(path).get_pixels(oiio.FLOAT)
    h, w = image.shape[:2]
    texels = np.zeros((h, w, 3))
    texels[..., :min(3, image.shape[2])] = image[..., :3]
    if control != "st origin at the top":
        texels = texels[::-1]
    if control == "st wrapped, not clamped":
        modes = ("periodic", "periodic")
    f = st * [w, h] - 0.5
    if control == "nearest texel":
        f = np.floor(st * [w, h])
    i0 = np.floor(f).astype(int)
    fx, fy = (f - i0)[:, :1], (f - i0)[:, 1:]

    def address(i, n, mode):
        if mode == "clamp":
            return np.clip(i, 0, n - 1)
        if mode == "periodic":
            return i % n
        raise ValueError(f"the reference samples clamp and periodic, not {mode}")

    x0, x1 = address(i0[:, 0], w, modes[0]), address(i0[:, 0] + 1, w, modes[0])
    y0, y1 = address(i0[:, 1], h, modes[1]), address(i0[:, 1] + 1, h, modes[1])
    return ((texels[y0, x0] * (1 - fx) + texels[y0, x1] * fx) * (1 - fy) +
            (texels[y1, x0] * (1 - fx) + texels[y1, x1] * fx) * fy)


def triangles(stage, time, host, control=None):
    """Every drawn mesh instance's fan triangles, in world space: for each,
    the world points, the object normals at its corners, the normal matrix,
    whether it is left-handed, a face id, and its material's emission with
    the sampled primvar at its corners (None for clay)."""
    out = []
    for instance, (prim, world) in enumerate(meshes(stage, time, host.purposes, control)):
        mesh = UsdGeom.Mesh(prim)
        emitted = emission(prim, host, control)
        if emitted and emitted != PINK:
            primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar(emitted[2])
            if not primvar or mesh.GetSubdivisionSchemeAttr().Get(time) != UsdGeom.Tokens.none:
                raise ValueError(f"{prim.GetPath()}: the reference samples an authored primvar "
                                 "of an unsubdivided mesh")
            st_values, st_interpolation = primvar.ComputeFlattened(time), primvar.GetInterpolation()
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

            def value(values, kind, c):
                if kind == UsdGeom.Tokens.constant:
                    return values[0]
                if kind == UsdGeom.Tokens.uniform:
                    return values[face]
                if kind == UsdGeom.Tokens.faceVarying:
                    return values[corner + c]
                return values[indices[corner + c]]

            for k in range(1, count - 1):
                cs = (0, k, k + 1)
                out.append({
                    "points": [world.Transform(Gf.Vec3d(points[indices[corner + c]])) for c in cs],
                    "object": [Gf.Vec3d(points[indices[corner + c]]) for c in cs],
                    "normals": [Gf.Vec3d(value(normals, interpolation, c)) for c in cs],
                    "normal_matrix": normal_matrix,
                    "left": left,
                    "centre": centre,
                    "face": (str(prim.GetPath()), instance, face),
                    "emission": emitted,
                    "st": ([list(value(st_values, st_interpolation, c)) for c in cs]
                           if emitted and emitted != PINK else None),
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


def reference(stage, time, width, height, host, control=None):
    focal = 1.005 if control == "focal length +0.5 %" else 1.0
    policy = CameraUtil.Crop if control == "aperture cropped" else CameraUtil.Fit
    f = frustum(stage, time, width, height, focal, policy)
    to_clip = f.ComputeViewMatrix() * f.ComputeProjectionMatrix()
    eye = np.array(f.position)
    if control == "orientations misread":
        misread_orientations(stage, time)
    tris = triangles(stage, time, host, control)
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
    if not tris:
        return image
    world = np.array([[list(v) for v in t["points"]] for t in tris])
    nearest, p, bary = cast(to_clip, width, height, shift, world)
    hit = nearest >= 0
    i = nearest[hit]
    p, bary = p[hit], bary[hit]

    # Row vectors, as Gf multiplies them: v @ m.
    m = np.array([[list(r) for r in t["normal_matrix"]] for t in tris])[i]
    o = np.array([[list(v) for v in t["object"]] for t in tris])[i]
    front = np.einsum("ni,nij->nj", np.cross(o[:, 1] - o[:, 0], o[:, 2] - o[:, 0]), m)
    front[np.array([t["left"] for t in tris])[i]] *= -1
    toward = normalize(eye - p)
    if control == "flat normals":
        n = normalize(front)
    else:
        normals = np.array([[list(n) for n in t["normals"]] for t in tris])[i]
        n = normalize(np.einsum("ni,nij->nj", np.einsum("nc,ncj->nj", bary, normals), m))
    behind = dot(front, toward) < 0
    if control != "unflipped normals":
        n[behind] *= -1
    if control == "flat normals":
        n[dot(n, toward) < 0] *= -1
    # Every mesh is clay until textures: Lambert at 18 % grey under a
    # headlight of π along the camera's axis, as src/surface.rs sets it, so
    # a facing surface shows its albedo. MaterialX's BSDFs turn a normal
    # facing away from the eye toward it before lighting.
    lit = n.copy()
    lit[dot(lit, toward) < 0] *= -1
    light = toward if control == "headlight from the eye" else -np.array(f.ComputeViewDirection())
    grey = 0.18 * np.clip(dot(lit, light), 0.0, 1.0)
    color = np.c_[grey, grey, grey, np.ones(len(grey))]
    pink = np.array([t["emission"] == PINK for t in tris])[i]
    color[pink, :3] = [1, 0, 1]
    kinds = sorted({t["emission"] for t in tris if t["emission"] and t["emission"] != PINK})
    which = np.array([kinds.index(t["emission"]) if t["emission"] in kinds else -1 for t in tris])[i]
    for k, emitted in enumerate(kinds):
        on = which == k
        st = np.array([t["st"] or [[0, 0]] * 3 for t in tris])[i][on]
        color[on, :3] = sample(emitted[0], np.einsum("nc,ncj->nj", bary[on], st), emitted[1], control)
    clip = np.c_[p, np.ones(len(p))] @ np.array(to_clip)
    depth = (clip[:, 2] / clip[:, 3] + 1) / 2
    if control == "unconverted depth":
        depth = 1 - depth
    faces = {}
    face = np.array([faces.setdefault(t["face"], len(faces)) for t in tris])
    image["color"][hit] = color
    image["depth"][hit] = depth
    image["normal"][hit] = n
    image["face"][hit] = face[i]
    return image


def cast(to_clip, width, height, shift, tris):
    """Casts a ray through each pixel centre at the triangles (world points,
    (count, 3, 3)): the nearest triangle's index or -1, the hit point and
    its barycentric coordinates, each per pixel. As Gf.Ray.Intersect from
    Gf.Frustum.ComputePickRay, a ray starts on the near plane, hits either
    side of a triangle, counts an edge as in, and sees nothing behind it."""
    # Barycentric slack: a ray exactly along an edge (the image's centre
    # line over a box's edge) otherwise misses both triangles by rounding.
    EDGE = 1e-9
    nearest = np.full((height, width), -1)
    point = np.zeros((height, width, 3))
    bary = np.zeros((height, width, 3))
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    unproject = np.linalg.inv(np.array(to_clip))
    columns = np.arange(width)
    ndc_x = 2 * (columns + 0.5 + shift) / width - 1
    # A row at a time: Moller-Trumbore over every (pixel, triangle) pair.
    for y in range(height):
        ndc_y = np.full(width, 1 - 2 * (y + 0.5) / height)
        near = np.c_[ndc_x, ndc_y, -np.ones(width), np.ones(width)] @ unproject
        far = np.c_[ndc_x, ndc_y, np.ones(width), np.ones(width)] @ unproject
        o = (near[:, :3] / near[:, 3:])[:, None]
        d = (far[:, :3] / far[:, 3:])[:, None] - o
        with np.errstate(divide="ignore", invalid="ignore"):
            pv = np.cross(d, e2)
            det = dot(e1, pv)
            s = o - v0
            u = dot(s, pv) / det
            qv = np.cross(s, e1)
            v = dot(d, qv) / det
            t = dot(e2, qv) / det
            hit = (det != 0) & (u >= -EDGE) & (v >= -EDGE) & (u + v <= 1 + EDGE) & (t >= 0)
        i = np.where(hit, t, np.inf).argmin(axis=1)
        seen = hit[columns, i]
        x, i = columns[seen], i[seen]
        nearest[y, x] = i
        point[y, x] = o[x, 0] + t[x, i, None] * d[x, 0]
        bary[y, x] = np.c_[1 - u[x, i] - v[x, i], u[x, i], v[x, i]]
    return nearest, point, bary


def dot(a, b):
    return (a * b).sum(axis=-1)


def normalize(v):
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, length, out=np.zeros_like(v), where=length > 0)


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


def references(stage, width, height, host):
    """The reference and each control's, for each frame."""
    controls = [None] + [c for c in CONTROLS if c != "frame 1's reference"]
    return {t: {c: reference(stage, t, width, height, host, c) for c in controls} for t in FRAMES}


def differs(stage, host):
    """Whether `host`'s reference can differ from husk's at the same size:
    by a purpose authored, or a material it does not deliver."""
    prims = list(stage.Traverse())
    purposed = any(UsdGeom.Imageable(p).GetPurposeAttr().HasAuthoredValue() for p in prims)
    dropped = any(p.IsA(UsdShade.Shader) and unknown(UsdShade.Shader(p)) for p in prims)
    return purposed or (dropped and host.delivers_unknown != HUSK.delivers_unknown)


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
        refs = references(stage, width, height, HUSK)
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
        host = usdrecord_host()
        if shape[:2] != (height, width) or differs(stage, host):
            refs = references(stage, shape[1], shape[0], host)
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
