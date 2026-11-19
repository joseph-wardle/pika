"""Pika's image check: renders each scene in hydra/images/ and compares the images with a ray cast
from USD. husk's color, depth and normals are compared where there is a husk; usdrecord's color
is compared when asked for. Each control perturbs the reference and must then fail.

    hython hydra/image.py <out dir> [usdrecord]

Run it from hydra/check.sh's environment; usdrecord needs a display.
"""

import collections
import concurrent.futures
import math
import os
import shutil
import subprocess
import sys

import MaterialX as mx
import numpy as np
import OpenImageIO as oiio
from pxr import CameraUtil, Gf, Sdf, Tf, Usd, UsdGeom, UsdImagingGL, UsdLux, UsdRender, UsdShade

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = os.path.join(ROOT, "hydra", "images")
SETTINGS = "/render/settings"
# The layers the scenes sublayer, which are not scenes themselves.
LAYERS = ("settings.usda", "light.usda")
# The delegate's default texture cap.
CAP = float("inf")
# Scenes whose texture the cap shrinks.
CAPPED = {"capped": 1}
# Scenes under PIKA_BUDGET.
BUDGET = {"lowered": 1}
TAIL = 4
# Scenes husk draws coarser than the camera needs, first frame and last, (0, 0) for the rest.
BLURRED = {"lowered": (0, 1)}
# Scenes whose pika:shadowPool holds fewer pages than husk's camera marks.
POOLED = {"receding": 1}
# Scenes whose shadow pages husk must show cached, or not.
CACHED = {"moving": True, "lamps": False}
# Scenes husk must draw the same twice, bit for bit.
REPEATED = {"receding"}
# Each render is a Houdini process of about 1 GB with a few busy threads.
RENDERS = 8
# Each reference process holds up to 1.4 GB (Houdini 22's hython).
REFERENCES = 16
# Pika's pixel filter (src/film.rs).
FILTER_RADIUS = 1.5
FILTER_SIGMA = 0.284
# Rays to a pixel's side in a textured scene's reference.
RAYS = 3
# A jump between adjacent rays of a textured scene's colour past which its pixels are not compared.
STEP = 0.1
# Scenes whose shadows thin below a pixel between the centres, which their references trace as a
# textured scene's.
THIN = {"receding"}
# Color allows for usdrecord's half floats.
TOLERANCE = {"color": 2e-3, "depth": 2e-6, "normal": 2e-4}
# Penumbra pixels' bands of the reference's share of their light, the pixels a band needs to be
# compared, and how far Pika's mean share over one may be from the reference's.
BANDS = ((0, 0.1), (0.1, 0.4), (0.4, 0.6), (0.6, 0.9), (0.9, 1))
BAND_PIXELS = 16
PENUMBRA = 0.15
CONTROLS = ["focal length +0.5 %", "half a pixel right", "aperture cropped", "unconverted depth",
            "unfiltered", "filter of half the radius", "depth and normals off centre",
            "unflipped normals", "flat normals", "instancer transform dropped", "outer instances dropped",
            "orientations misread", "authored normals on subdivided meshes", "holes drawn",
            "face normals averaged", "left-handed corners unswapped", "frame 1's smooth normals",
            "cage drawn on subdivided meshes", "boundary rules of the interior",
            "purposes ignored", "culled by box centre", "camera light from the eye",
            "usdview's camera light in every host", "camera light held as the other host holds it",
            "Pika's old headlight", "intensity read as lux",
            "sphere light read as a point of its intensity", "sphere light's scale ignored",
            "lights shining up their z axis", "area lights read as points", "area lights turned 90°",
            "area lights' scale ignored", "area lights lit from both sides", "light links ignored",
            "instance links ignored",
            "shadows ignored", "penumbrae wholly shadowed", "shadow links ignored", "shadow:enable ignored", "shadow rays turned 6°",
            "shadow rays past local lights", "moved casters' old shadows kept",
            "shadows of the light's centre alone", "shadows of a quarter of the light's size",
            "st origin at the top", "st wrapped, not clamped", "nearest texel",
            "useMetadata read as periodic", "file wrap metadata ignored",
            "st read whatever the name", "UV0 drawn pink", "st seams smoothed over", "st refined bilinearly",
            "faceVarying st corners kept apart",
            "failed materials drawn clay", "base weight ignored", "emission ignored",
            "normal map ignored", "tangent from the first axis", "bitangent from the winding",
            "subsets ignored", "subset faces of the cage", "st indices ignored",
            "sRGB read as raw", "8-bit auto read as raw", "8-bit data read as sRGB", "ACEScg primaries kept",
            "UDIM tiles read as one image", "UDIM rows one tile wide", "missing UDIM tiles read as 1001",
            "level 0 whatever the footprint", "the nearest level, not a blend", "levels filtered from level 0",
            "the cap ignored", "lowering ignored", "missing file's fallback ignored", "a zero mapped normal left unlit",
            "constant nodes at their default",
            "frame 1's reference"]
# The controls under which a subdivided mesh is drawn as its cage.
CAGE_DRAWN = ("cage drawn on subdivided meshes", "authored normals on subdivided meshes")
# ACEScg to linear Rec.709.
ACESCG_TO_REC709 = np.array([[1.705051, -0.621792, -0.083259],
                             [-0.130256, 1.140805, -0.010548],
                             [-0.024003, -0.128969, 1.152972]])
# A material's closed form.
Look = collections.namedtuple("Look", "albedo emission image normal facing")
CLAY = Look((0.18,) * 3, (0.0,) * 3, None, None, None)
PINK = Look((0.0,) * 3, (1.0, 0.0, 1.0), None, None, None)
# Each lit surface's inputs.
LIT = {
    "ND_open_pbr_surface_surfaceshader": ("base_weight", "base_color", "emission_luminance",
                                          "emission_color", "specular_weight", "geometry_normal", 35),
    "ND_standard_surface_surfaceshader": ("base", "base_color", "emission", "emission_color",
                                          "specular", "normal", None),
}
# What a host hands Pika.
Host = collections.namedtuple("Host", "purposes delivers_unknown settings camera_light holds_camera_light")
# husk's camera light, /husk_headlight/__the_headlight, is not usdview's, and husk holds it where
# the camera was at the first frame it renders.
HUSK = Host(("default", "render"), True, True, 57225, True)
# usdview's and usdrecord's, from HdxTaskController.
USDRECORD_CAMERA_LIGHT = 15000
CAMERA_LIGHT_ANGLE = 0.53
# The scenes rendered under the host's camera light.
CAMERA_LIT = {"headlit"}


def usdrecord_host():
    scene_index = Tf.GetEnvSetting("USDIMAGINGGL_ENGINE_ENABLE_SCENE_INDEX")
    return Host(("default", "proxy"), scene_index, False, USDRECORD_CAMERA_LIGHT, False)


def meshes(stage, time, purposes, control=None):
    """Every mesh of `purposes` drawn, once per instance it is drawn at."""
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
    """Authors, in the session layer, each point instancer's orientations as they would read if
    GfQuatf's (i, j, k, real) were taken for (real, i, j, k)."""
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
    """One level of uniform refinement."""
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
    out_counts, out_indices, out_holes, out_parents = [], [], [], []
    for f, face in enumerate(faces):
        middle = len(refined)
        refined.append(centre[f])
        for j, v in enumerate(face):
            after = index[frozenset((v, face[(j + 1) % len(face)]))]
            before = index[frozenset((face[j - 1], v))]
            if f in holes:
                out_holes.append(len(out_counts))
            out_counts.append(4)
            out_parents.append(f)
            child = [v, after, middle, before]
            # OpenSubdiv rotates a quad's children to keep its parametric orientation, which
            # decides the diagonal each child is split on.
            rotation = j if len(face) == 4 else 0
            out_indices += child[-rotation:] + child[:-rotation] if rotation else child
    return refined, out_counts, out_indices, set(out_holes), out_parents


def surface(mesh, time, control=None):
    """The points, face vertex counts, face vertex indices and holes a mesh is drawn with at
    `time`, and each drawn face's cage face."""
    points = mesh.GetPointsAttr().Get(time)
    counts = mesh.GetFaceVertexCountsAttr().Get(time)
    indices = mesh.GetFaceVertexIndicesAttr().Get(time)
    holes = set(mesh.GetHoleIndicesAttr().Get(time) or [])
    scheme = mesh.GetSubdivisionSchemeAttr().Get(time)
    if scheme == UsdGeom.Tokens.none or control in CAGE_DRAWN:
        return points, counts, indices, holes, list(range(len(counts)))
    if scheme not in (UsdGeom.Tokens.catmullClark, UsdGeom.Tokens.bilinear):
        raise ValueError(f"{mesh.GetPath()}: the reference does not refine {scheme}")
    authored = [mesh.GetCreaseIndicesAttr(), mesh.GetCornerIndicesAttr()]
    if any(a.HasAuthoredValue() for a in authored) or (
            mesh.GetInterpolateBoundaryAttr().Get(time) != UsdGeom.Tokens.edgeAndCorner):
        raise ValueError(f"{mesh.GetPath()}: the reference refines only uncreased edgeAndCorner meshes")
    return refine(points, counts, indices, holes, scheme, control)


def weld(counts, indices, values, control=None):
    """Welds a faceVarying primvar's values by point and value, as Pika does."""
    welded, index, corner_value = [], {}, []
    for point, value in zip(indices, values):
        key = point if control == "st seams smoothed over" else (point, tuple(value))
        if key not in index:
            index[key] = len(welded)
            welded.append(value)
        corner_value.append(index[key])
    # Each edge's sides.
    sides = {}
    corner = 0
    for count in counts:
        for j in range(count):
            a, b = corner + j, corner + (j + 1) % count
            side = {indices[a]: corner_value[a], indices[b]: corner_value[b]}
            sides.setdefault(frozenset((indices[a], indices[b])), []).append(side)
        corner += count
    parted = collections.Counter()
    for edge, faces in sides.items():
        if len(faces) == 2 and faces[0] == faces[1]:
            continue
        if len(faces) == 2 and any(faces[0][v] == faces[1][v] for v in edge):
            raise ValueError("the reference does not refine a seam ending inside the surface")
        parted.update(edge)
    if any(n > 2 for n in parted.values()):
        raise ValueError("the reference does not refine a vertex where three seams or boundaries meet")
    return welded, corner_value


def texcoord(mesh, primvar, time, control=None):
    """The primvar as the faces drawn read it."""
    values, interpolation = primvar.ComputeFlattened(time), primvar.GetInterpolation()
    if control == "st indices ignored":
        values = primvar.Get(time)
    scheme = mesh.GetSubdivisionSchemeAttr().Get(time)
    if scheme == UsdGeom.Tokens.none or control in CAGE_DRAWN:
        return values, interpolation
    counts = mesh.GetFaceVertexCountsAttr().Get(time)
    indices = mesh.GetFaceVertexIndicesAttr().Get(time)
    holes = set(mesh.GetHoleIndicesAttr().Get(time) or [])
    if interpolation == UsdGeom.Tokens.faceVarying:
        rule = mesh.GetFaceVaryingLinearInterpolationAttr().Get(time)
        if rule != UsdGeom.Tokens.cornersPlus1:
            raise ValueError(f"{mesh.GetPath()}: the reference refines faceVarying primvars under cornersPlus1 only")
        values, indices = weld(counts, indices, values, control)
    elif interpolation != UsdGeom.Tokens.vertex:
        raise ValueError(f"{mesh.GetPath()}: the reference refines vertex and faceVarying primvars only")
    # Storm's reading.
    if control == "st refined bilinearly":
        scheme = UsdGeom.Tokens.bilinear
    sheet = [Gf.Vec3d(v[0], v[1], 0) for v in values]
    refined, _, corners, _, _ = refine(sheet, counts, indices, holes, scheme, control)
    return [Gf.Vec2f(refined[c][0], refined[c][1]) for c in corners], UsdGeom.Tokens.faceVarying


def upstream(shader, name):
    """The shader connected to `shader`'s input `name`, or None."""
    source = shader.GetInput(name).GetConnectedSource() if shader.GetInput(name) else None
    return UsdShade.Shader(source[0].GetPrim()) if source else None


def nodedefs():
    """The host's MaterialX libraries, as the generate tool loads them."""
    libraries = mx.createDocument()
    search = mx.FileSearchPath(os.environ["MATERIALX"])
    mx.loadLibraries(["libraries"], search, libraries)
    return libraries


NODEDEFS = nodedefs()


def unknown(shader):
    """Whether the shader names a nodedef the libraries do not define."""
    return shader.GetShaderId().startswith("ND_") and not NODEDEFS.getNodeDef(shader.GetShaderId())


def uv0(image):
    """Whether the image node samples UV0, the texcoord MaterialX's nodes read by default."""
    if not image.GetInput("texcoord"):
        return True
    node = upstream(image, "texcoord")
    if not node or node.GetShaderId() != "ND_texcoord_vector2":
        return False
    index = node.GetInput("index")
    return not index or index.Get() == 0


def sampled(image, kind, time, control=None):
    """The file, address modes (u, v) at `time`, primvar and declared colour space (auto when none
    for a colour, raw for the rest) of an ND_image_<kind> node sampled at UV0 (st) or a
    geompropvalue reader, or of a UsdUVTexture (`kind` rgb, its output) sampled at st or a
    UsdPrimvarReader_float2, or None for any other node."""
    usd = kind == "rgb"
    if not image or image.GetShaderId() != ("UsdUVTexture" if usd else f"ND_image_{kind}"):
        return None
    if usd:
        reader = upstream(image, "st")
        if reader and reader.GetShaderId() != "UsdPrimvarReader_float2":
            return None
        primvar = reader.GetInput("varname").Get() if reader else "st"
    elif uv0(image):
        primvar = "st"
    else:
        reader = upstream(image, "texcoord")
        if not reader or reader.GetShaderId() != "ND_geompropvalue_vector2":
            return None
        primvar = reader.GetInput("geomprop").Get()
    file = image.GetInput("file")
    # USD resolves a path to the file it names, which a <UDIM> pattern is not.
    path = file.Get().resolvedPath
    if "<UDIM>" in file.Get().path:
        path = file.GetAttr().GetPropertyStack(Usd.TimeCode.Default())[0].layer.ComputeAbsolutePath(file.Get().path)
    colour = usd or kind == "color3" or control == "8-bit data read as sRGB"
    space = file.GetAttr().GetColorSpace() or ("auto" if colour else "raw")
    if not path:
        # A file that does not resolve is sampled nowhere (`preview`).
        return (path, (), primvar, space)
    if usd:
        modes = tuple(wrap(image, name, path, time, control) for name in ("wrapS", "wrapT"))
    else:
        modes = tuple(image.GetInput(m).Get(time) if image.GetInput(m) else "periodic"
                      for m in ("uaddressmode", "vaddressmode"))
    return (path, modes, primvar, space)


def wrap(image, name, path, time, control=None):
    """A UsdUVTexture's wrapS or wrapT at `time` in the image nodedefs' words."""
    token = image.GetInput(name).Get(time) if image.GetInput(name) else "useMetadata"
    if token != "useMetadata":
        return "periodic" if token == "repeat" else token
    if control == "useMetadata read as periodic":
        return "periodic"
    return metadata(path, control)[name == "wrapT"]


def metadata(path, control=None):
    """The wrap modes (u, v) the file's wrapmodes attribute states, as maketx writes it
    ("black,clamp", or one word for both), black for an axis it states none or another for, as
    Storm reads it."""
    if "<UDIM>" in path:
        path = next(p for p in (path.replace("<UDIM>", str(n)) for n in range(1001, 1101)) if os.path.isfile(p))
    file = oiio.ImageInput.open(path)
    stated = "" if control == "file wrap metadata ignored" else file.spec().getattribute("wrapmodes") or ""
    file.close()
    words = stated.split(",")
    return tuple(w if w in ("black", "clamp", "periodic", "mirror") else "black" for w in (words + words)[:2])


def constant(shader, name):
    """The constant node `shader`'s input `name` reads through dot nodes, or None."""
    node = upstream(shader, name)
    while node and node.GetShaderId().startswith("ND_dot_"):
        node = upstream(node, "in")
    return node if node and node.GetShaderId().startswith("ND_constant_") else None


def lit(surface, time, control=None):
    """The Look of an OpenPBR or Standard Surface shader with its specular off
    (hydra/images/pbr.usda), its inputs read at `time` as values or constant nodes' values
    (hydra/images/constant.usda), its normal mapped when connected
    (hydra/images/normalmap.usda)."""
    *names, bump, facing = LIT[surface.GetShaderId()]
    path = surface.GetPath()
    authored = {i.GetBaseName(): i for i in surface.GetInputs()}
    extra = sorted(set(authored) - set(names) - {bump})
    connected = [n for n, i in authored.items()
                 if i.HasConnectedSource() and n != bump and not constant(surface, n)]
    if extra or connected:
        raise ValueError(f"{path}: the reference lights a surface's base and emission as values; "
                         f"it has no closed form for {extra or connected}")
    normal = None
    if bump in authored:
        node = upstream(surface, bump)
        chain = node and node.GetShaderId() == "ND_normalmap_float" and (
            [i.GetBaseName() for i in node.GetInputs()] == ["in"])
        normal = sampled(upstream(node, "in"), "vector3", time, control) if chain else None
        if not normal:
            raise ValueError(f"{path}: the reference perturbs a normal only by a normalmap of an "
                             "image's vector3 sample at a primvar")

    def value(name):
        node = constant(surface, name) if name in authored else None
        shader, name = (node, "value") if node else (surface, name)
        input = shader.GetInput(name)
        default = node and control == "constant nodes at their default"
        if input and input.GetAttr().HasAuthoredValue() and not default:
            v = input.Get(time)
        else:
            v = NODEDEFS.getNodeDef(shader.GetShaderId()).getActiveInput(name).getValue()
        return tuple(v) if hasattr(v, "__len__") else v

    if value(names[4]) != 0:
        raise ValueError(f"{path}: the reference has no closed form for a specular lobe; set {names[4]} to 0")
    weight = 1.0 if control == "base weight ignored" else value(names[0])
    glow = 0.0 if control == "emission ignored" else value(names[2])
    return Look(tuple(weight * c for c in value(names[1])), tuple(glow * c for c in value(names[3])),
                None, normal, facing)


def preview(surface, time, control=None):
    """The Look of a UsdPreviewSurface (hydra/images/wrapped.usda)."""
    path = surface.GetPath()
    authored = {i.GetBaseName(): i for i in surface.GetInputs()}
    extra = sorted(set(authored) - {"diffuseColor", "emissiveColor", "useSpecularWorkflow", "specularColor",
                                    "normal"})
    connected = [n for n, i in authored.items()
                 if i.HasConnectedSource() and n not in ("emissiveColor", "normal")]
    if extra or connected:
        raise ValueError(f"{path}: the reference lights a UsdPreviewSurface's diffuseColor and emissiveColor "
                         f"as values; it has no closed form for {extra or connected}")

    def value(name):
        if name in authored:
            v = authored[name].Get(time)
        else:
            v = NODEDEFS.getNodeDef("ND_UsdPreviewSurface_surfaceshader").getActiveInput(name).getValue()
        return tuple(v) if hasattr(v, "__len__") else v

    if value("useSpecularWorkflow") != 1 or value("specularColor") != (0.0, 0.0, 0.0):
        raise ValueError(f"{path}: the reference has no closed form for a UsdPreviewSurface's specular lobe; "
                         "set useSpecularWorkflow to 1 and specularColor to black")
    albedo = value("diffuseColor")
    if "normal" in authored:
        node = upstream(surface, "normal")
        source = node and sampled(node, "rgb", time, control)
        fallback = node and node.GetInput("fallback")
        if not source or source[0] or (fallback and tuple(fallback.Get(time))[:3] != (0.0, 0.0, 0.0)):
            raise ValueError(f"{path}: the reference maps a UsdPreviewSurface's normal only by a UsdUVTexture "
                             "whose file does not resolve and whose fallback is the zero vector")
        if control == "a zero mapped normal left unlit":
            albedo = (0.0,) * 3
    image, emission = None, value("emissiveColor")
    if "emissiveColor" in authored and authored["emissiveColor"].HasConnectedSource():
        node = upstream(surface, "emissiveColor")
        image = sampled(node, "rgb", time, control)
        if not image:
            raise ValueError(f"{path}: the reference emits a UsdUVTexture's rgb sampled at a primvar, "
                             "nothing else connected")
        emission = (0.0,) * 3
        if not image[0]:
            fallback = node.GetInput("fallback")
            if fallback and control != "missing file's fallback ignored":
                emission = tuple(fallback.Get(time))[:3]
            image = None
    return Look(albedo, emission, image, None, 30)


def look(prim, host, time, control=None):
    """The closed form of the mesh's material."""
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not material:
        return CLAY
    source = material.GetSurfaceOutput().GetConnectedSource()
    surface = UsdShade.Shader(source[0].GetPrim()) if source else None
    refuse = ValueError(f"{prim.GetPath()}: the reference knows only clay, pink, lit surfaces and the emissive ramp")
    if surface and unknown(surface):
        clay = control == "failed materials drawn clay" or not host.delivers_unknown
        return CLAY if clay else PINK
    if surface and surface.GetShaderId() == "UsdPreviewSurface":
        return preview(surface, time, control)
    if surface and surface.GetShaderId() in LIT:
        return lit(surface, time, control)
    if not surface or surface.GetShaderId() != "ND_surface":
        raise refuse
    edf = upstream(surface, "edf")
    if not edf or edf.GetShaderId() != "ND_uniform_edf" or upstream(surface, "bsdf"):
        raise refuse
    node = upstream(edf, "color")
    while node and node.GetShaderId().startswith("ND_convert_"):
        node = upstream(node, "in")
    image = (sampled(node, "vector2", time, control) or sampled(node, "vector3", time, control)
             or sampled(node, "color3", time, control))
    if not image:
        raise refuse
    if control == "UV0 drawn pink" and uv0(node):
        return PINK
    return Look((0.0,) * 3, (0.0,) * 3, image, None, None)


def working(texels, path, space, control=None):
    """The texels in the working space, linear Rec.709, from the colour space the asset declares."""
    if space == "auto":
        eight = oiio.ImageBuf(path).spec().format.basetype == oiio.UINT8
        space = "srgb_texture" if eight and control != "8-bit auto read as raw" else "raw"
    elif control == "sRGB read as raw" and space == "srgb_texture":
        space = "raw"
    elif control == "ACEScg primaries kept" and space == "acescg":
        space = "raw"
    if space == "srgb_texture":
        return np.where(texels <= 0.04045, texels / 12.92, ((texels + 0.055) / 1.055) ** 2.4)
    if space == "acescg":
        return texels @ ACESCG_TO_REC709.T
    if space != "raw":
        raise ValueError(f"{path}: the reference reads srgb_texture, acescg, raw and auto, not {space}")
    return texels


def halve(texels):
    """A 2×2 box of the texels (rows, columns, channels), the last row or column repeated where a
    side is odd, as the texture tool builds a level."""
    h, w = texels.shape[:2]
    padded = texels[np.minimum(np.arange(h + h % 2), h - 1)][:, np.minimum(np.arange(w + w % 2), w - 1)]
    return padded.reshape((h + 1) // 2, 2, (w + 1) // 2, 2, -1).mean((1, 3))


def bilinear(texels, st, modes, control=None):
    """The bilinear sample of one level's texels, rows from the bottom, at each st (an (n, 2)
    array), under the address modes (u, v)."""
    h, w = texels.shape[:2]
    f = st * [w, h] - 0.5
    if control == "nearest texel":
        f = np.floor(st * [w, h])
    i0 = np.floor(f).astype(int)
    fx, fy = (f - i0)[:, :1], (f - i0)[:, 1:]

    def address(i, n, mode):
        """The texel column or row `i` reads, in `padded`, whose border of 0 is what black
        addresses past the edge."""
        if mode == "black":
            return np.clip(i, -1, n) + 1
        if mode == "clamp":
            return np.clip(i, 0, n - 1) + 1
        if mode == "periodic":
            return i % n + 1
        raise ValueError(f"the reference samples black, clamp and periodic, not {mode}")

    padded = np.pad(texels, ((1, 1), (1, 1), (0, 0)))
    x0, x1 = address(i0[:, 0], w, modes[0]), address(i0[:, 0] + 1, w, modes[0])
    y0, y1 = address(i0[:, 1], h, modes[1]), address(i0[:, 1] + 1, h, modes[1])
    return ((padded[y0, x0] * (1 - fx) + padded[y0, x1] * fx) * (1 - fy) +
            (padded[y1, x0] * (1 - fx) + padded[y1, x1] * fx) * fy)


def sample(path, st, step, modes, space, cap, control=None):
    """The trilinear sample of the image at `path`, or of the tile of the UDIM set `path` names, at
    each st (an (n, 2) array), rows from the bottom as USD reads st, under the address modes (u,
    v), in the working space."""
    if "<UDIM>" in path:
        tile = np.floor(st)
        if control == "UDIM tiles read as one image":
            tile[:] = 0
        width = 1 if control == "UDIM rows one tile wide" else 10
        number = 1001 + tile[:, 0].astype(int) + width * tile[:, 1].astype(int)
        out = np.zeros((len(st), 3))
        for n in np.unique(number):
            file = path.replace("<UDIM>", str(n))
            if not os.path.isfile(file) and control == "missing UDIM tiles read as 1001":
                file = path.replace("<UDIM>", "1001")
            if os.path.isfile(file):
                on = number == n
                out[on] = sample(file, (st - tile)[on], step[on], modes, space, cap, control)
        return out
    if control == "st wrapped, not clamped":
        modes = ("periodic", "periodic")
    if control == "the cap ignored":
        cap = float("inf")
    sizes = []
    file = oiio.ImageInput.open(path)
    while file.seek_subimage(0, len(sizes)):
        sizes.append((file.spec().width, file.spec().height))
    file.close()
    first = next((m for m, (w, h) in enumerate(sizes) if w <= cap and h <= cap), None)
    if first is None:
        raise ValueError(f"{path}: the reference uses a stored level within the cap, and none of its {len(sizes)} is within {cap}")
    level0 = oiio.ImageBuf(path).get_pixels(oiio.FLOAT)

    def level(m):
        """Level m's texels, the file's level first + m, in the working space, rows from the
        bottom."""
        if first + m and control == "levels filtered from level 0":
            image = level0
            for _ in range(first + m):
                image = halve(image)
        else:
            image = oiio.ImageBuf(path, 0, first + m).get_pixels(oiio.FLOAT)
            if image.size == 0:
                raise ValueError(f"{path}: the reference reads the levels the file stores, which stop before level {first + m}")
        texels = np.zeros(image.shape[:2] + (3,))
        texels[..., :min(3, image.shape[2])] = image[..., :3]
        texels = working(texels, path, space, control)
        return texels if control == "st origin at the top" else texels[::-1]

    w, h = sizes[first]
    with np.errstate(divide="ignore"):
        lod = np.clip(np.log2(np.hypot(step[..., 0] * w, step[..., 1] * h).max(1)), 0, None)
    if control == "level 0 whatever the footprint":
        lod[:] = 0
    if control == "the nearest level, not a blend":
        lod = np.floor(lod + 0.5)
    lower = np.floor(lod).astype(int)
    f = lod - lower
    out = np.zeros((len(st), 3))
    for m in np.unique(np.r_[lower, lower[f > 0] + 1]):
        weight = np.where(lower == m, 1 - f, np.where(lower + 1 == m, f, 0))
        on = weight > 0
        out[on] += weight[on, None] * bilinear(level(m), st[on], modes, control)
    return out


def tangent_sums(points, counts, indices, holes, key, st):
    """Lengyel's sums over the drawn triangles around each vertex."""
    u = collections.defaultdict(lambda: np.zeros(3))
    v = collections.defaultdict(lambda: np.zeros(3))
    corner = 0
    for face, count in enumerate(counts):
        for k in range(1, count - 1) if face not in holes else ():
            tri = (corner, corner + k, corner + k + 1)
            for j in range(3):
                a, b, c = tri[j], tri[(j + 1) % 3], tri[(j + 2) % 3]
                p = [np.array(points[indices[i]]) for i in (a, b, c)]
                e1, e2 = p[1] - p[0], p[2] - p[0]
                d1, d2 = np.subtract(st(b), st(a)), np.subtract(st(c), st(a))
                det = d1[0] * d2[1] - d2[0] * d1[1]
                if det == 0:
                    continue
                u[key(a)] += (e1 * d2[1] - e2 * d1[1]) / det
                v[key(a)] += (e2 * d1[0] - e1 * d2[0]) / det
        corner += count
    return u, v


def frame(normal, u, v, control=None):
    """A vertex's unit tangent and handedness from its normal and summed u and v directions."""
    n = normalize(np.array(normal, float))
    t = u - n * np.dot(n, u)
    if not t.any() or control == "tangent from the first axis":
        t = np.cross((0, 1, 0) if abs(n[1]) < 0.9 else (1, 0, 0), n)
    t = normalize(t) if t.any() else np.array((1.0, 0.0, 0.0))
    mirrored = control != "bitangent from the winding" and np.dot(np.cross(n, t), v) < 0
    return t, -1.0 if mirrored else 1.0


def parts(prim, parents, host, time, control=None):
    """The Look of each drawn face."""
    own = look(prim, host, time, control)
    out = [None] * len(parents)
    subsets = UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets()
    for subset in () if control == "subsets ignored" else subsets:
        bound = look(subset.GetPrim(), host, time, control)
        faces = set(subset.GetIndicesAttr().Get(time))
        for drawn, cage in enumerate(parents):
            face = drawn if control == "subset faces of the cage" else cage
            if face in faces and out[drawn] is None:
                out[drawn] = bound
    return [own if part is None else part for part in out]


def triangles(stage, time, host, control=None):
    """Every drawn mesh instance's fan triangles, in world space."""
    out = []
    for instance, (prim, world) in enumerate(meshes(stage, time, host.purposes, control)):
        mesh = UsdGeom.Mesh(prim)
        points, counts, indices, holes, parents = surface(mesh, time, control)
        looks = parts(prim, parents, host, time, control)
        images = [l.image or l.normal for l in looks if l.image or l.normal]
        names = sorted({i[2] for i in images})
        if len(names) > 1:
            raise ValueError(f"{prim.GetPath()}: its materials sample {names}; a mesh is drawn with one texcoord")
        image = images[0] if images else None
        if image:
            name = "st" if control == "st read whatever the name" else image[2]
            primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar(name)
            if not primvar:
                raise ValueError(f"{prim.GetPath()}: the reference samples an authored primvar")
            st_values, st_interpolation = texcoord(mesh, primvar, time, control)
        # Gf multiplies row vectors, so normals go through the inverse transpose.
        linear = world.ExtractRotationMatrix()
        normal_matrix = linear.GetInverse().GetTranspose()
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
            at = stage.GetStartTimeCode() if control == "frame 1's smooth normals" else time
            normals = smooth_normals(surface(mesh, at, control)[0], counts, indices, left, control)
            interpolation = UsdGeom.Tokens.vertex
        face_of = [face for face, count in enumerate(counts) for _ in range(count)]

        def value(values, kind, c):
            """The value corner `c` reads, by the primvar's interpolation."""
            if kind == UsdGeom.Tokens.constant:
                return values[0]
            if kind == UsdGeom.Tokens.uniform:
                return values[face_of[c]]
            if kind == UsdGeom.Tokens.faceVarying:
                return values[c]
            return values[indices[c]]

        frames = {}
        if any(l.normal for l in looks):
            def key(c):
                st = tuple(value(st_values, st_interpolation, c))
                if (control == "faceVarying st corners kept apart" and not subdivided
                        and st_interpolation == UsdGeom.Tokens.faceVarying):
                    st = c
                return indices[c], tuple(value(normals, interpolation, c)), st

            u, v = tangent_sums(points, counts, indices, holes, key,
                                lambda c: value(st_values, st_interpolation, c))

            def frame_of(c):
                k = key(c)
                if k not in frames:
                    frames[k] = frame(k[1], u[k], v[k], control)
                return frames[k]
        corner = 0
        for face, count in enumerate(counts):
            for k in range(1, count - 1) if face not in holes else ():
                cs = (corner, corner + k, corner + k + 1)
                out.append({
                    "points": [world.Transform(Gf.Vec3d(points[indices[c]])) for c in cs],
                    "object": [Gf.Vec3d(points[indices[c]]) for c in cs],
                    "normals": [Gf.Vec3d(value(normals, interpolation, c)) for c in cs],
                    "normal_matrix": normal_matrix,
                    "linear": linear,
                    "left": left,
                    "centre": centre,
                    "face": (str(prim.GetPath()), instance, face),
                    "look": looks[face],
                    "st": [list(value(st_values, st_interpolation, c)) for c in cs] if image else None,
                    "frames": [frame_of(c) for c in cs] if looks[face].normal else None,
                })
            corner += count
    return out


def illuminance(intensity, exposure, color, angle, normalize, control):
    """What a distant light gives a surface facing it."""
    theta = min(max(math.radians(angle) / 2, 0), math.pi)
    sin2 = math.sin(theta) ** 2
    if theta == 0:
        received = size = 1.0
    elif theta <= math.pi / 2:
        received = size = math.pi * sin2
    else:
        received, size = math.pi, (2 - sin2) * math.pi
    if not normalize:
        size = 1.0
    if control == "intensity read as lux":
        received = size = 1.0
    return tuple(np.array(color) * intensity * 2 ** exposure * received / size)


def sphere(api, radius, time, p, control):
    """The direction toward a sphere light from each hit at `p`, the illuminance it gives a surface
    there facing it and the sine of the angle it subtends."""
    m = UsdGeom.Xformable(api.GetPrim()).ComputeLocalToWorldTransform(time)
    scales = [np.linalg.norm([m[r][0], m[r][1], m[r][2]]) for r in range(3)]
    if max(scales) - min(scales) > 1e-6 * max(scales):
        raise ValueError(f"{api.GetPath()} is scaled unevenly: an ellipsoid has no closed form")
    r = radius * (1 if control == "sphere light's scale ignored" else scales[0])
    toward = np.array(m.ExtractTranslation()) - p
    d = np.linalg.norm(toward, axis=1)
    if (d <= r).any():
        raise ValueError(f"a surface is inside {api.GetPath()}")
    radiance = np.array(api.GetColorAttr().Get(time)) * api.GetIntensityAttr().Get(time) * 2 ** api.GetExposureAttr().Get(time)
    if api.GetNormalizeAttr().Get(time):
        radiance = radiance / (4 * np.pi * r ** 2)
    sin2 = (r / d) ** 2
    if control == "sphere light read as a point of its intensity":
        sin2 = 1 / (np.pi * d ** 2)
    return toward / d[:, None], np.pi * sin2[:, None] * radiance, r / d


def lambert(corners, c):
    """The illuminance vector of an outline of radiance 1, `corners` (hits, corners, 3) relative to
    each hit, pointing toward `c`, the light's centre relative to each hit (Lambert's formula)."""
    b = np.roll(corners, -1, axis=1)
    k = np.cross(corners, b)
    s = np.linalg.norm(k, axis=2)
    gamma = np.arctan2(s, np.einsum("nkj,nkj->nk", corners, b))
    e = 0.5 * np.einsum("nk,nkj->nj", np.where(s > 0, gamma / np.where(s > 0, s, 1), 0), k)
    e[dot(e, c) < 0] *= -1
    return e


def area(prim, time, p, n, control):
    """The direction a rect, disk or cylinder light shines on each hit at `p`, facing `n`, and the
    illuminance it gives a surface there facing that way."""
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)
    x, y, z, o = (np.array([m[r][0], m[r][1], m[r][2]]) for r in range(4))
    cylinder = prim.IsA(UsdLux.CylinderLight)
    if control == "area lights turned 90°":
        # A cylinder about its z axis, keeping its axis as far from a surface below it as it was.
        k = z if cylinder else y
        k = k / np.linalg.norm(k)
        x, y, z = (np.cross(k, v) + np.dot(k, v) * k for v in (x, y, z))
    if control == "area lights' scale ignored":
        x, y, z = (v / np.linalg.norm(v) for v in (x, y, z))
    api = UsdLux.LightAPI(prim)
    radiance = np.array(api.GetColorAttr().Get(time)) * api.GetIntensityAttr().Get(time) * 2 ** api.GetExposureAttr().Get(time)
    c = o - p
    lit = np.ones(len(p), bool)
    if cylinder:
        light = UsdLux.CylinderLight(prim)
        length, radius = light.GetLengthAttr().Get(time), light.GetRadiusAttr().Get(time)
        ys, zs = np.linalg.norm(y), np.linalg.norm(z)
        if abs(ys - zs) > 1e-6 * ys:
            raise ValueError(f"{prim.GetPath()} scales y and z unevenly: an elliptic cylinder has no outline here")
        r, half = radius * ys, x * length / 2
        size = 2 * np.pi * r * length * np.linalg.norm(x)
        shown = size / np.pi
        axis = x / np.linalg.norm(x)
        out = np.outer(dot(c, axis), axis) - c
        rho = np.linalg.norm(out, axis=1)
        lit = rho > r
        out /= rho[:, None]
        t = np.arccos(np.clip(r / rho, -1, 1))[:, None] * np.linspace(1, -1, 64)
        arc = r * (np.cos(t)[..., None] * out[:, None] + np.sin(t)[..., None] * np.cross(axis, out)[:, None])
        corners = np.concatenate([(c + half)[:, None] + arc, ((c - half)[:, None] + arc)[:, ::-1]], axis=1)
        emitted = np.ones(len(p))
    else:
        if prim.IsA(UsdLux.RectLight):
            light = UsdLux.RectLight(prim)
            w, h = light.GetWidthAttr().Get(time), light.GetHeightAttr().Get(time)
            outline = [(a * w / 2, b * h / 2) for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
            size = w * h * np.linalg.norm(np.cross(x, y))
        else:
            r = UsdLux.DiskLight(prim).GetRadiusAttr().Get(time)
            # The regular 64-gon of the disk's area.
            k = 64
            s = r * math.sqrt(2 * math.pi / (k * math.sin(2 * math.pi / k)))
            outline = [(s * math.cos(2 * math.pi * i / k), s * math.sin(2 * math.pi * i / k)) for i in range(k)]
            size = math.pi * r ** 2 * np.linalg.norm(np.cross(x, y))
        shown = size
        corners = c[:, None] + np.array([a * x + b * y for a, b in outline])
        front = np.cross(x, y)
        front *= -np.sign(np.dot(front, z)) / np.linalg.norm(front)
        # The cosine at the light of the direction to each hit.
        emitted = np.clip(dot(-c, front) / np.linalg.norm(c, axis=1), 0, None)
        if control != "area lights lit from both sides":
            lit = emitted > 0
    if api.GetNormalizeAttr().Get(time):
        radiance = radiance / size
    e = lambert(corners, c)
    if control == "area lights read as points":
        d = np.linalg.norm(c, axis=1)
        e = c * (shown * emitted / d ** 3)[:, None]
    e[~lit] = 0
    seen = lit & (dot(e, n) > 0)
    if (np.einsum("nkj,nj->nk", corners, n)[seen] < 0).any():
        raise ValueError(f"a surface sees {prim.GetPath()} cross its horizon: only a whole outline has a closed form")
    magnitude = np.linalg.norm(e, axis=1)
    direction = np.where(magnitude[:, None] > 0, e / np.where(magnitude > 0, magnitude, 1)[:, None], n)
    return direction, magnitude[:, None] * radiance, 0


AREA = (UsdLux.RectLight, UsdLux.DiskLight, UsdLux.CylinderLight)


def links(prim, paths, control):
    """Which hits' meshes, by the hits' prim `paths`, the light's lightLink collection includes."""
    if control == "light links ignored":
        return np.ones(len(paths), bool)
    query = UsdLux.LightAPI(prim).GetLightLinkCollectionAPI().ComputeMembershipQuery()
    return member(prim, query, paths, control)


def member(light, query, paths, control):
    """Which `paths` the collection `query` includes."""
    stage = light.GetStage()
    everything = dict(query.GetAsPathExpansionRuleMap()) == {Sdf.Path.absoluteRootPath: "expandPrims"}
    def included(path):
        proxy = stage.GetPrimAtPath(path).IsInstanceProxy()
        if control == "instance links ignored" and proxy and not everything:
            return False
        return query.IsPathIncluded(Sdf.Path(path))
    included = {path: included(path) for path in set(paths)}
    return np.array([included[path] for path in paths], bool)


# Hit-triangle pairs a shadow ray test holds at once, about 100 MB.
PAIRS = 1 << 20


def shadowed(toward, casts, p, own, world, control, reach=None):
    """Which hits `p`, on triangles `own`, a light toward `toward` (one direction, or one per hit)
    shadows."""
    if control == "shadows ignored":
        return np.zeros(len(p), bool)
    if control == "shadow rays turned 6°":
        a = math.radians(6)
        side = normalize(np.cross(toward, (0.0, 1.0, 0.0)))
        toward = normalize(toward * math.cos(a) + side * math.sin(a))
    k = np.nonzero(casts)[0]
    v0, e1, e2 = world[k, 0], world[k, 1] - world[k, 0], world[k, 2] - world[k, 0]
    out = np.zeros(len(p), bool)
    chunk = max(1, PAIRS // max(len(k), 1))
    for start in range(0, len(p), chunk):
        d = toward if toward.ndim == 1 else toward[start:start + chunk, None]
        u, v, t, det = intersect(p[start:start + chunk, None], d, v0, e1, e2)
        hit = (det != 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
        if reach is not None and control != "shadow rays past local lights":
            hit &= t < reach[start:start + chunk, None]
        out[start:start + chunk] = (hit & (k != own[start:start + chunk, None])).any(axis=1)
    return out


# Where a sized light's shadow rays aim, over [0, 1]².
GRID = (np.stack(np.meshgrid(np.arange(8), np.arange(8)), -1).reshape(-1, 2) + 0.5) / 8
SIDE_EDGE = np.array([(x, y) for y in (0, 0.5, 1) for x in (0, 0.5, 1) if (x, y) != (0.5, 0.5)])
DISK_EDGE = np.c_[np.ones(8), np.arange(8) / 8]


def aims(prim, time, toward, angle, control):
    """Where a light's shadow rays aim."""
    if control == "shadows of the light's centre alone":
        return None
    scale = 0.25 if control == "shadows of a quarter of the light's size" else 1.0

    def disk(u):
        return scale * np.sqrt(u[:, :1]) * np.c_[np.cos(2 * np.pi * u[:, 1]), np.sin(2 * np.pi * u[:, 1])]

    def side(u):
        return scale * (2 * u - 1)

    if toward is not None:
        t = math.tan(min(math.radians(angle) / 2, math.radians(80)))
        if t == 0:
            return None
        e0 = normalize(np.cross(toward, (0.0, 1.0, 0.0) if abs(toward[1]) < 0.9 else (1.0, 0.0, 0.0)))
        e1 = np.cross(toward, e0)
        return (lambda u, p: normalize(toward + t * (disk(u)[:, :1] * e0 + disk(u)[:, 1:] * e1))), DISK_EDGE
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)
    x, y, z, c = (np.array([m[r][0], m[r][1], m[r][2]]) for r in range(4))
    if prim.IsA(UsdLux.SphereLight):
        r = UsdLux.SphereLight(prim).GetRadiusAttr().Get(time) * np.linalg.norm(x)
        if r == 0:
            return None

        def sphere_disk(u, p):
            w = normalize(c - p)
            a = normalize(np.cross(w, np.where(np.abs(w[:, :1]) < 0.9, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))))
            b = np.cross(w, a)
            d = disk(u)
            return c + r * (d[:, 0, None, None] * a + d[:, 1, None, None] * b)
        return sphere_disk, DISK_EDGE
    if prim.IsA(UsdLux.RectLight):
        light = UsdLux.RectLight(prim)
        half = np.array([x * light.GetWidthAttr().Get(time) / 2, y * light.GetHeightAttr().Get(time) / 2])
        return (lambda u, p: np.broadcast_to((c + side(u) @ half)[:, None], (len(u), len(p), 3))), SIDE_EDGE
    if prim.IsA(UsdLux.DiskLight):
        r = UsdLux.DiskLight(prim).GetRadiusAttr().Get(time)
        return (lambda u, p: np.broadcast_to((c + disk(u) @ np.array([x * r, y * r]))[:, None], (len(u), len(p), 3))), DISK_EDGE
    light = UsdLux.CylinderLight(prim)
    half = x * light.GetLengthAttr().Get(time) / 2
    r = light.GetRadiusAttr().Get(time) * np.linalg.norm(y)
    axis = half / np.linalg.norm(half)

    def cylinder_side(u, p):
        out = normalize(np.cross(axis, p - c))
        s = side(u)
        return c + s[:, 0, None, None] * half + s[:, 1, None, None] * r * out
    return cylinder_side, SIDE_EDGE


def share(prim, time, toward, angle, p, own, world, casts, control):
    """Each hit's share of a light."""
    distant = toward is not None

    def unmet(points, at):
        """The share of the hits `at`'s rays toward `points`, directions (n, 3) or points per hit
        (n, hits at, 3), that meet no caster."""
        blocked = np.zeros(at.sum())
        for q in points:
            if distant:
                blocked += shadowed(q, casts, p[at], own[at], world, control)
            else:
                v = q - p[at]
                d = np.linalg.norm(v, axis=1)
                blocked += shadowed(v / d[:, None], casts, p[at], own[at], world, control, d)
        return 1 - blocked / len(points)

    every = np.ones(len(p), bool)
    if distant:
        centre = toward[None]
    else:
        o = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time).ExtractTranslation()
        centre = np.broadcast_to(np.array(o), (1, len(p), 3))
    found = aims(prim, time, toward, angle, control)
    if found is None:
        return unmet(centre, every)
    at, edge = found
    out = unmet(np.concatenate([centre, at(edge, p)]), every)
    split = (out > 0) & (out < 1)
    if split.any():
        out[split] = unmet(at(GRID, p[split]), split)
    return out


def shadow_casters(prim, faces, control):
    """Which triangles, by `faces`' prim paths, cast the light's shadows."""
    if prim.GetAttribute("inputs:shadow:enable").Get() is False and control != "shadow:enable ignored":
        return np.zeros(len(faces), bool)
    if control == "shadow links ignored":
        return np.ones(len(faces), bool)
    query = UsdLux.LightAPI(prim).GetShadowLinkCollectionAPI().ComputeMembershipQuery()
    return member(prim, query, faces, control)


def lights(stage, time, axis, toward, p, n, paths, host, lit, casters, control=None):
    """Each light as the direction toward it, one for every hit or one per hit, the illuminance it
    gives a surface facing it unshadowed, likewise, the sine of the angle it subtends, 0 but for
    a sphere, and each hit's share of it its shadow leaves."""
    own, world, faces = casters
    found = [prim for prim in stage.Traverse() if prim.HasAPI(UsdLux.LightAPI)]
    for prim in found:
        if not any(prim.IsA(t) for t in (UsdLux.DistantLight, UsdLux.SphereLight, *AREA)):
            raise ValueError(f"{prim.GetPath()} is a {prim.GetTypeName()}: only distant, sphere, rect, disk and cylinder lights have a closed form")
    if not found and lit:
        light = toward if control == "camera light from the eye" else -axis
        intensity = USDRECORD_CAMERA_LIGHT if control == "usdview's camera light in every host" else host.camera_light
        e = illuminance(intensity, 0, (1, 1, 1), CAMERA_LIGHT_ANGLE, False, control)
        light = np.asarray(light, float)
        # From the eye, a ray per hit along its own direction.
        angle = CAMERA_LIGHT_ANGLE if light.ndim == 1 else 0
        left = share(None, time, light, angle, p, own, world, np.ones(len(world), bool), control)
        return [(light, e, 0, left)]
    if not found and control == "Pika's old headlight":
        return [(-axis, (np.pi,) * 3, 0, np.ones(len(p)))]
    out = []
    for prim in found:
        if UsdGeom.Imageable(prim).ComputeVisibility(time) == UsdGeom.Tokens.invisible:
            continue
        api = UsdLux.LightAPI(prim)
        on = links(prim, paths, control)
        left = np.ones(len(p))
        local = prim.IsA(UsdLux.SphereLight) or any(prim.IsA(t) for t in AREA)
        if prim.IsA(UsdLux.SphereLight):
            given = sphere(api, UsdLux.SphereLight(prim).GetRadiusAttr().Get(time), time, p[on], control)
        elif local:
            given = area(prim, time, p[on], n[on], control)
        if local:
            left[on] = share(prim, time, None, 0, p[on], own[on], world, shadow_casters(prim, faces, control), control)
        else:
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)
            z = np.array([m[2][0], m[2][1], m[2][2]])
            if control == "lights shining up their z axis":
                z = -z
            e = illuminance(api.GetIntensityAttr().Get(time), api.GetExposureAttr().Get(time),
                            api.GetColorAttr().Get(time), UsdLux.DistantLight(prim).GetAngleAttr().Get(time),
                            api.GetNormalizeAttr().Get(time), control)
            given = (z / np.linalg.norm(z), e, 0)
            casts = shadow_casters(prim, faces, control)
            angle = UsdLux.DistantLight(prim).GetAngleAttr().Get(time)
            left[on] = share(None, time, given[0], angle, p[on], own[on], world, casts, control)
        d, e, sin = np.zeros((len(p), 3)), np.zeros((len(p), 3)), np.zeros(len(p))
        d[on], e[on], sin[on] = given
        out.append((d, e, sin, left))
    return out


def frustum(stage, time, width, height, focal, policy):
    settings = UsdRender.Settings(stage.GetPrimAtPath(SETTINGS))
    camera_path = settings.GetCameraRel().GetTargets()[0]
    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path)).GetCamera(time)
    camera.focalLength *= focal
    f = camera.frustum
    CameraUtil.ConformWindow(f, policy, width / height)
    return f


def reference(stage, time, width, height, host, lowered, lit, thin, control=None, centre=None, base=None):
    """The reference image of the stage at `time`."""
    if control is None and centre is not None:
        image = dict(centre)
    else:
        image = trace(stage, time, width, height, 1, host, lowered, lit,
                      None if control == "depth and normals off centre" else control)
    if control == "depth and normals off centre":
        off = trace(stage, time, width, height, 1, host, lowered, lit, control)
        image["depth"], image["normal"] = off["depth"], off["normal"]
    radius = FILTER_RADIUS / 2 if control == "filter of half the radius" else FILTER_RADIUS
    same = base is not None and radius == FILTER_RADIUS and all(
        np.array_equal(image[k], centre[k]) for k in ("color", "face", "shadow", "bounds", "cell"))
    traced = (image["textured"] or thin) and control != "unfiltered"
    if traced and same:
        for k in ("color", "bounds", "soft", "mixed"):
            image[k] = base[k]
        return image
    if traced:
        fine = trace(stage, time, width * RAYS, height * RAYS, RAYS, host, lowered, lit, control)
        filtered(image, fine, radius)
    elif base is not None and all(np.array_equal(image[k], centre[k]) for k in ("face", "shadow")):
        image["mixed"] = base["mixed"]
        image["bounds"] = image["bounds"][image["soft"]]
        return image
    else:
        # The geometry alone, at the textured scenes' rays.
        fine = {"face": coverage(stage, time, width, height, host, control)}
        image["mixed"] = mixed(fine, image, ("face",))
    # Pika's shadow edges stray up to a map texel, a pixel, from the rays', and the filter spreads
    # them further.
    image["mixed"] |= edges(image["shadow"], 2)
    # An edge pixel sees a sliver past its outermost rays, and nothing past the image.
    image["mixed"][[0, -1], :] = True
    image["mixed"][:, [0, -1]] = True
    # Bounds only where they differ.
    image["bounds"] = image["bounds"][image["soft"]]
    return image


def weight(radius, distance2):
    """Pika's filter's weight at `distance2` pixels squared from a pixel's centre."""
    r = distance2 / radius ** 2
    return np.where(r < 1, np.exp(-0.5 * r / FILTER_SIGMA ** 2), 0.0)


def windows(a, kernel):
    """Each of `kernel`'s weights with the rays of `a`, traced RAYS × RAYS a pixel, at its offset
    from every pixel's centre, the edge pixels' rays repeated past the image's edge."""
    h, w = a.shape[0] // RAYS, a.shape[1] // RAYS
    rest = a.shape[2:]
    cells = a.reshape((h, RAYS, w, RAYS) + rest)
    cells = np.pad(cells, [(1, 1), (0, 0), (1, 1), (0, 0)] + [(0, 0)] * len(rest), mode="edge")
    padded = cells.reshape(((h + 2) * RAYS, (w + 2) * RAYS) + rest)
    for i, j in zip(*np.nonzero(kernel)):
        yield kernel[i, j], padded[i:i + h * RAYS:RAYS, j:j + w * RAYS:RAYS]


def kernel(radius):
    """The filter's weight at each ray of a pixel and its neighbours each way."""
    d = (np.arange(3 * RAYS) + 0.5) / RAYS - 1.5
    return weight(radius, d[:, None] ** 2 + d[None, :] ** 2)


def mixed(fine, centre, keys):
    """Pixels whose `keys` differ from `centre`'s at a ray of `fine` in the pixel or a neighbour."""
    out = np.zeros(centre[keys[0]].shape, bool)
    for key in keys:
        for _, v in windows(fine[key], np.ones((3 * RAYS, 3 * RAYS))):
            out |= v != centre[key]
    return out


def stepped(color):
    """Pixels whose filter meets two adjacent rays of `color`, traced RAYS × RAYS a pixel,
    differing by more than STEP."""
    h, w = color.shape[0] // RAYS, color.shape[1] // RAYS
    jump = np.zeros(color.shape[:2])
    across = np.abs(np.diff(color[..., :3], axis=1)).max(axis=-1)
    down = np.abs(np.diff(color[..., :3], axis=0)).max(axis=-1)
    jump[:, :-1] = np.maximum(jump[:, :-1], across)
    jump[:-1] = np.maximum(jump[:-1], down)
    out = np.zeros((h, w), bool)
    for _, v in windows(jump > STEP, np.ones((3 * RAYS, 3 * RAYS))):
        out |= v
    return out


def filtered(image, fine, radius):
    """Gives `image` the colour, penumbra bounds and penumbra pixels of `fine`, its trace at RAYS ×
    RAYS rays a pixel, filtered by `radius`."""
    k = kernel(radius)
    total = k.sum()
    image["color"] = sum(w * v for w, v in windows(fine["color"], k)) / total
    image["bounds"] = sum(w * v for w, v in windows(fine["bounds"], k)) / total
    image["soft"] = np.logical_or.reduce([v for _, v in windows(fine["soft"], k)])
    image["mixed"] = mixed(fine, image, ("face", "shadow", "cell")) | stepped(fine["color"])


# Controls that move the rays, by the fraction of a pixel.
SHIFTS = {"half a pixel right": 0.5, "depth and normals off centre": 0.25}


def lens(control):
    """The focal length's factor and the window policy under `control`."""
    focal = 1.005 if control == "focal length +0.5 %" else 1.0
    policy = CameraUtil.Crop if control == "aperture cropped" else CameraUtil.Fit
    return focal, policy


def view(stage, time, width, height, host, control):
    """The camera's frustum at `time`, conformed to a `width` × `height` image, its world to clip
    matrix, and the triangles drawn."""
    f = frustum(stage, time, width, height, *lens(control))
    to_clip = f.ComputeViewMatrix() * f.ComputeProjectionMatrix()
    if control == "orientations misread":
        misread_orientations(stage, time)
    tris = triangles(stage, time, host, control)
    stage.GetSessionLayer().Clear()
    if control == "culled by box centre":
        def seen(p):
            c = to_clip.Transform(p)
            return f.GetNearFar().Contains(-f.ComputeViewMatrix().Transform(p)[2]) and all(abs(v) <= 1 for v in (c[0], c[1]))
        tris = [t for t in tris if seen(t["centre"])]
    return f, to_clip, tris


def face_ids(tris):
    """Each triangle's face as a number, the same for every trace of the same triangles."""
    faces = {}
    return np.array([faces.setdefault(t["face"], len(faces)) for t in tris])


def coverage(stage, time, width, height, host, control=None):
    """The face each of RAYS × RAYS rays a pixel meets, -1 for none, at RAYS times the image's
    size."""
    _, to_clip, tris = view(stage, time, width * RAYS, height * RAYS, host, control)
    out = np.full((height * RAYS, width * RAYS), -1)
    if not tris:
        return out
    world = np.array([[list(v) for v in t["points"]] for t in tris])
    nearest, _, _ = cast(to_clip, width * RAYS, height * RAYS, SHIFTS.get(control, 0.0) * RAYS, world)
    out[nearest >= 0] = face_ids(tris)[nearest[nearest >= 0]]
    return out


def trace(stage, time, width, height, scale, host, lowered, lit, control=None):
    """The stage at `time` traced through each pixel centre of a `width` × `height` image, `scale`
    times as fine as the reference's."""
    f, to_clip, tris = view(stage, time, width, height, host, control)
    fixed = host.holds_camera_light != (control == "camera light held as the other host holds it")
    focal, policy = lens(control)
    axis = np.array((frustum(stage, stage.GetStartTimeCode(), width, height, focal, policy) if fixed else f).ComputeViewDirection())
    authored = stage.GetPrimAtPath(SETTINGS).GetAttribute("pika:textureCap")
    # 0 is none, as it is to Pika.
    cap = (authored.Get() if host.settings and authored else 0) or CAP
    if lowered and control != "lowering ignored":
        cap = min(cap, TAIL)
    eye = np.array(f.position)
    shift = SHIFTS.get(control, 0.0) * scale

    image = {
        "color": np.zeros((height, width, 4)),
        "depth": np.ones((height, width)),
        "normal": np.zeros((height, width, 3)),
        "face": np.full((height, width), -1),
        "shadow": np.zeros((height, width), int),
        # Penumbra pixels, and at each, row by row, its colour with its partly shadowed lights
        # wholly shadowed and wholly lit.
        "soft": np.zeros((height, width), bool),
        "bounds": np.zeros((height, width, 2, 3)),
        # Per pixel, the step of its look's image it lies in (`sampled`), 0 for none.
        "cell": np.zeros((height, width), np.int64),
        "textured": False,
    }
    if not tris:
        return image
    world = np.array([[list(v) for v in t["points"]] for t in tris])
    faces = [t["face"][0] for t in tris]
    casting = world
    if control == "moved casters' old shadows kept" and time != stage.GetStartTimeCode():
        # The triangles that moved since frame 1, where they were then.
        old = triangles(stage, stage.GetStartTimeCode(), host)
        moved = old if len(old) != len(tris) else [o for o, t in zip(old, tris) if o["points"] != t["points"]]
        if moved:
            casting = np.concatenate([world, np.array([[list(v) for v in o["points"]] for o in moved])])
            faces = faces + [o["face"][0] for o in moved]
    nearest, p, bary = cast(to_clip, width, height, shift, world)
    around = footprint(to_clip, width, height, shift, world, nearest, scale)
    hit = nearest >= 0
    i = nearest[hit]
    p, bary = p[hit], bary[hit]

    # Row vectors, as Gf multiplies them.
    m = np.array([[list(r) for r in t["normal_matrix"]] for t in tris])[i]
    o = np.array([[list(v) for v in t["object"]] for t in tris])[i]
    front = np.einsum("ni,nij->nj", np.cross(o[:, 1] - o[:, 0], o[:, 2] - o[:, 0]), m)
    front[np.array([t["left"] for t in tris])[i]] *= -1
    toward = normalize(eye - p)
    normals = np.array([[list(n) for n in t["normals"]] for t in tris])[i]
    if control == "flat normals":
        n = normalize(front)
    else:
        n = normalize(np.einsum("ni,nij->nj", np.einsum("nc,ncj->nj", bary, normals), m))
    shade = n.copy()
    behind = dot(front, toward) < 0
    if control != "unflipped normals":
        n[behind] *= -1
    if control == "flat normals":
        n[dot(n, toward) < 0] *= -1
    looks = [t["look"] for t in tris]
    image["textured"] = any(l.image or l.normal for l in looks)

    def sampled(field):
        """Each hit's sample of its look's `field` image at its st, 0 where it has none, and which
        hits have one."""
        out = np.zeros((len(p), 3))
        images = sorted({getattr(l, field) for l in looks if getattr(l, field)})
        which = np.array([images.index(getattr(l, field)) if getattr(l, field) else -1 for l in looks])[i]
        for k, (path, modes, _, space) in enumerate(images):
            on = which == k
            st = np.array([t["st"] or [[0, 0]] * 3 for t in tris])[i][on]
            at = np.einsum("nc,ncj->nj", bary[on], st)
            step = np.einsum("nkc,ncj->nkj", around[on], st) - at[:, None]
            out[on] = sample(path, at, step, modes, space, cap, control)
            # A sample steps where st crosses into another UDIM tile, or across an image's edge
            # into a black border.
            cell = np.floor(at).astype(np.int64)
            for axis, mode in enumerate(modes):
                if mode != "black" and "<UDIM>" not in path:
                    cell[:, axis] = 0
            cells[np.flatnonzero(on)] = cell[:, 0] * 65536 + cell[:, 1]
        return out, which >= 0

    cells = np.zeros(len(p), np.int64)
    # A normal map perturbs the normal before it is turned toward the eye.
    value, mapped = sampled("normal")
    if mapped.any() and control != "normal map ignored":
        value = np.where(dot(value, value)[:, None] == 0, (0.0, 0.0, 1.0), value * 2 - 1)
        frames = [t["frames"] or [((0, 0, 0), 1.0)] * 3 for t in tris]
        tangents = np.array([[list(t) for t, _ in f] for f in frames])[i]
        handed = np.array([[w for _, w in f] for f in frames])[i]
        # Each corner's bitangent, from its unnormalized object normal, as the vertex stage
        # computes it.
        bitangents = handed[:, :, None] * np.cross(normals, tangents)
        r = np.array([[list(row) for row in t["linear"]] for t in tris])[i]
        t = normalize(np.einsum("ni,nij->nj", np.einsum("nc,ncj->nj", bary, tangents), r))
        b = normalize(np.einsum("ni,nij->nj", np.einsum("nc,ncj->nj", bary, bitangents), r))
        perturbed = normalize(t * value[:, :1] + b * value[:, 1:2] + shade * value[:, 2:])
        shade[mapped] = perturbed[mapped]
    # MaterialX's BSDFs turn a normal facing away from the eye toward it before lighting.
    facing = shade
    facing[dot(facing, toward) < 0] *= -1
    shine, low, high = np.zeros((3, len(p), 3))
    paths = [tris[k]["face"][0] for k in i]
    # Two bits a light.
    shadow = np.zeros(len(p), int)
    found = lights(stage, time, axis, toward, p, facing, paths, host, lit, (i, casting, faces), control)
    for bit, (d, e, sin, left) in enumerate(found):
        partly = (left > 0) & (left < 1)
        shadow |= ((left == 0) * 2 + partly) << (2 * bit)
        cos = dot(facing, d)
        if (abs(cos) < sin).any():
            raise ValueError("a surface sees a sphere light cross its horizon: only a whole sphere has a closed form")
        given = np.clip(cos, 0.0, 1.0)[:, None] * np.array(e) / np.pi
        shine += given * left[:, None]
        low += given * np.where(partly, 0, left)[:, None]
        high += given * np.where(partly, 1, left)[:, None]
    limit = np.array([180 if l.facing is None else l.facing for l in looks])[i]
    if (dot(facing, toward) < np.cos(np.radians(limit))).any():
        raise ValueError("a surface is seen from further round than the angle its closed form holds within")
    albedo = np.array([l.albedo for l in looks])[i]
    emission = np.array([l.emission for l in looks])[i]
    emitted = emission + sampled("image")[0]
    color = np.c_[albedo * shine + emitted, np.ones(len(p))]
    clip = np.c_[p, np.ones(len(p))] @ np.array(to_clip)
    depth = (clip[:, 2] / clip[:, 3] + 1) / 2
    if control == "unconverted depth":
        depth = 1 - depth
    face = face_ids(tris)
    image["color"][hit] = color
    image["cell"][hit] = cells
    image["depth"][hit] = depth
    image["normal"][hit] = n
    image["face"][hit] = face[i]
    image["shadow"][hit] = shadow
    bounds = image["bounds"]
    bounds[hit] = np.stack([albedo * low + emitted, albedo * high + emitted], 1)
    image["soft"][hit] = (high > low).any(axis=1)
    if control == "penumbrae wholly shadowed":
        # Their bounds meet, which leaves no share to compare.
        image["color"][..., :3] = np.where(image["soft"][..., None], bounds[..., 0, :], image["color"][..., :3])
        bounds[..., 1, :] = bounds[..., 0, :]
    return image


def rays(unproject, width, height, x, y):
    """The ray through the image position (x, y), in pixels, rows down."""
    ndc = np.c_[2 * x / width - 1, 1 - 2 * y / height]
    near = np.c_[ndc, -np.ones(len(x)), np.ones(len(x))] @ unproject
    far = np.c_[ndc, np.ones(len(x)), np.ones(len(x))] @ unproject
    o = near[:, :3] / near[:, 3:]
    return o, far[:, :3] / far[:, 3:] - o


def intersect(o, d, v0, e1, e2):
    """Moller-Trumbore."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pv = np.cross(d, e2)
        det = dot(e1, pv)
        s = o - v0
        u = dot(s, pv) / det
        qv = np.cross(s, e1)
        v = dot(d, qv) / det
        t = dot(e2, qv) / det
    return u, v, t, det


def cast(to_clip, width, height, shift, tris):
    """Casts a ray through each pixel centre at the triangles (world points, (count, 3, 3))."""
    # Barycentric slack.
    EDGE = 1e-9
    nearest = np.full((height, width), -1)
    point = np.zeros((height, width, 3))
    bary = np.zeros((height, width, 3))
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    unproject = np.linalg.inv(np.array(to_clip))
    columns = np.arange(width)
    for y in range(height):
        o, d = rays(unproject, width, height, columns + 0.5 + shift, np.full(width, y + 0.5))
        o, d = o[:, None], d[:, None]
        u, v, t, det = intersect(o, d, v0, e1, e2)
        hit = (det != 0) & (u >= -EDGE) & (v >= -EDGE) & (u + v <= 1 + EDGE) & (t >= 0)
        i = np.where(hit, t, np.inf).argmin(axis=1)
        seen = hit[columns, i]
        x, i = columns[seen], i[seen]
        nearest[y, x] = i
        point[y, x] = o[x, 0] + t[x, i, None] * d[x, 0]
        bary[y, x] = np.c_[1 - u[x, i] - v[x, i], u[x, i], v[x, i]]
    return nearest, point, bary


def footprint(to_clip, width, height, shift, tris, nearest, scale):
    """For each pixel `cast` hit, in image order."""
    y, x = np.nonzero(nearest >= 0)
    i = nearest[y, x]
    unproject = np.linalg.inv(np.array(to_clip))
    out = np.zeros((len(x), 2, 3))
    for k, (dx, dy) in enumerate(((1, 0), (0, 1))):
        o, d = rays(unproject, width, height, x + 0.5 + shift + dx * scale, y + 0.5 + dy * scale)
        u, v, _, _ = intersect(o, d, tris[i, 0], tris[i, 1] - tris[i, 0], tris[i, 2] - tris[i, 0])
        out[:, k] = np.c_[1 - u - v, u, v]
    return out


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


def edges(face, reach=1):
    """Pixels whose neighbourhood `reach` pixels each way holds more than one face (or none)."""
    padded = np.pad(face, reach, mode="edge")
    h, w = face.shape
    out = np.zeros(face.shape, bool)
    for dy in range(2 * reach + 1):
        for dx in range(2 * reach + 1):
            out |= padded[dy:dy + h, dx:dx + w] != face
    return out


def compare(image, ref, aovs, trusted=None):
    """Why `image` does not match `ref`, or an empty list, over the pixels `trusted` (`ref` by
    default) compares."""
    trusted = ref if trusted is None else trusted
    problems = []
    covered = ref["face"] >= 0
    edge = edges(trusted["face"])
    drawn = image["color"][..., 3] > 0.5
    stray = (drawn != covered) & ~edge
    if stray.any():
        problems.append(f"coverage differs at {stray.sum()} pixels away from edges")
    inner = covered & drawn & ~edge
    # Each of a still frame's samples rounds to a map texel, so a penumbra's ends stray a pixel
    # further than one ray's.
    shadow = edges(trusted["shadow"]) | (edges(trusted["shadow"], 2) & edges(trusted["soft"], 2))
    lit = inner & ~shadow & ~ref["soft"] & ~trusted["mixed"]
    for aov in aovs:
        error = np.abs(image[aov][..., :4] - ref[aov][..., :4] if aov == "color" else image[aov] - ref[aov])
        allowed = TOLERANCE[aov]
        if aov == "color":
            allowed = allowed + np.abs(ref[aov][..., :4]) / 256
        # A NaN is over.
        over = ~(error <= allowed)
        over = over.reshape(over.shape[:2] + (-1,)).any(-1) & (lit if aov == "color" else inner)
        if over.any():
            slack = " plus 1/256 of the value" if aov == "color" else ""
            problems.append(f"{aov} differs by up to {error[over].max():.3g} (tolerance {TOLERANCE[aov]:g}{slack})")
    return problems + penumbrae(image, ref, inner & ref["soft"])


def penumbrae(image, ref, at):
    """Why the penumbra pixels `at` of `image` do not match `ref`, or an
    empty list."""
    problems = []
    bounds = np.zeros(at.shape + (2, 3))
    bounds[ref["soft"]] = ref["bounds"]
    low, high = bounds[at][:, 0], bounds[at][:, 1]
    color = image["color"][at][:, :3]
    allowed = TOLERANCE["color"] + np.abs(high) / 256
    out = ((color < low - allowed) | (color > high + allowed)).any(axis=1)
    if out.any():
        problems.append(f"{out.sum()} penumbra pixels lie outside their lights wholly shadowed and wholly lit")
    span = (high - low).sum(axis=1)
    # Too little of the light to read its share from.
    seen = span > 30 * TOLERANCE["color"]
    shown = (color - low).sum(axis=1)[seen] / span[seen]
    share = (ref["color"][at][:, :3] - low).sum(axis=1)[seen] / span[seen]
    compared = 0
    for a, b in BANDS:
        band = (share >= a) & (share < b)
        if band.sum() < BAND_PIXELS:
            continue
        compared += 1
        if abs(shown[band].mean() - share[band].mean()) > PENUMBRA:
            problems.append(f"penumbra pixels of share {a}-{b} show {shown[band].mean():.2f} of their light, "
                            f"not {share[band].mean():.2f}")
    if seen.any() and not compared:
        problems.append(f"no band holds {BAND_PIXELS} of the {seen.sum()} penumbra pixels")
    return problems


def budget(name):
    """The environment a scene renders in."""
    env = dict(os.environ)
    if name in BUDGET:
        env["PIKA_BUDGET"] = str(BUDGET[name])
    return env


def times(stage):
    """The times of the stage's time range."""
    return tuple(range(int(stage.GetStartTimeCode()), int(stage.GetEndTimeCode()) + 1))


def husk(scene, out, times):
    name = os.path.splitext(os.path.basename(scene))[0]
    log = os.path.join(out, f"{name}.husk.log")
    env = budget(name)
    env["PIKA_STATS"] = os.path.join(out, f"{name}.husk.stats")
    headlight = [] if name in CAMERA_LIT else ["--headlight", "none"]
    with open(log, "w") as f:
        subprocess.run(["husk", "-V", "1", "-R", "Pika", "-s", SETTINGS, "--disable-motionblur", *headlight,
                        "-f", str(times[0]), "-n", str(len(times)),
                        "-o", os.path.join(out, f"{name}.husk.$F4.exr"), scene],
                       stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
    return {t: read(os.path.join(out, f"{name}.husk.{t:04d}.exr")) for t in times}, log


def stats(path):
    """Each frame line of the PIKA_STATS file at `path`, its values by name."""
    with open(path) as f:
        lines = [line.split() for line in f]
    return [dict(zip(w[2::2], w[3::2])) for w in lines if w and w[0] == "frame"]


def blurred(frames, name):
    """Failures unless husk's `frames` stats report BLURRED's textures drawn coarser than needed on
    the first frame and the last."""
    counts = [int(f["blurred_textures"]) for f in frames]
    expected = BLURRED.get(name, (0, 0))
    if not counts or (counts[0], counts[-1]) != expected:
        found = (counts[0], counts[-1]) if counts else "no frames"
        return [f"{name}: {found} textures drawn blurred on the first and last frames, not {expected}"]
    return []


def cached(frames, name):
    """Failures unless husk's `frames` stats show each time past the first keeping some of its
    shadow pages, drawing others anew and giving some a live copy, as a CACHED scene whose
    casters move must, and not, as one whose casters stay must."""
    if name not in CACHED:
        return []
    seen = []
    sample = 0
    for f in frames:
        if sample > 0 and int(f["sample"]) == 0:
            seen.append(tuple(int(f[k]) for k in ("shadow_pages_drawn", "shadow_pages_live", "shadow_pages_held")))
        sample = int(f["sample"])
    caches = all(0 < drawn < held and live > 0 for drawn, live, held in seen)
    if not seen or caches != CACHED[name]:
        return [f"{name}: shadow pages drawn, live and held at the times past the first {seen}"]
    return []


def repeated(name, renders, again):
    """Failures unless husk's second render, `again`, matches `renders` bit for bit."""
    def same(a, b):
        return all(np.array_equal(a[aov], b[aov]) for aov in a)
    failures = [f"{name} frame {t}: drawn again, the image differs" for t in sorted(renders)
                if not same(renders[t], again[t])]
    first, last = min(renders), max(renders)
    if same(renders[first], again[last]):
        failures.append(f"{name}: the control failed: frames {first} and {last} are the same")
    print(f"  drawn again: {'the same bit for bit' if not failures else 'differs'}")
    return failures


def usdrecord(scene, out, width, times):
    name = os.path.splitext(os.path.basename(scene))[0]
    log = os.path.join(out, f"{name}.usdrecord.log")
    off = [] if name in CAMERA_LIT else ["--disableCameraLight"]
    with open(log, "w") as f:
        subprocess.run(["usdrecord", "--renderer", "Pika", "--camera", "/camera", *off,
                        "--imageWidth", str(width), "--colorCorrectionMode", "disabled",
                        "--frames", f"{times[0]}:{times[-1]}",
                        scene, os.path.join(out, f"{name}.usdrecord.#.exr")],
                       stdout=f, stderr=subprocess.STDOUT, check=True, env=budget(name))
    return {t: read(os.path.join(out, f"{name}.usdrecord.{t}.exr")) for t in times}, log


def references(scene, width, height, host):
    """The reference and each control's, for each frame, of the scene at `scene`."""
    stage = Usd.Stage.Open(scene)
    name = os.path.splitext(os.path.basename(scene))[0]
    lowered, lit, thin = name in BUDGET, name in CAMERA_LIT, name in THIN
    controls = [c for c in CONTROLS if c != "frame 1's reference"]
    out = {}
    for t in times(stage):
        centre = trace(stage, t, width, height, 1, host, lowered, lit)
        base = reference(stage, t, width, height, host, lowered, lit, thin, None, centre)
        out[t] = {None: base} | {c: reference(stage, t, width, height, host, lowered, lit, thin, c, centre, base)
                                 for c in controls}
    return out


def differs(stage, host):
    """Whether `host`'s reference can differ from husk's at the same size."""
    prims = list(stage.Traverse())
    purposed = any(UsdGeom.Imageable(p).GetPurposeAttr().HasAuthoredValue() for p in prims)
    settings = any(a.GetName().startswith("pika:") for a in stage.GetPrimAtPath(SETTINGS).GetAttributes())
    dropped = any(p.IsA(UsdShade.Shader) and unknown(UsdShade.Shader(p)) for p in prims)
    return purposed or (settings and not host.settings) or (dropped and host.delivers_unknown != HUSK.delivers_unknown)


def budget_warnings(name):
    """The warnings of the memory budget scene `name` must print."""
    return int(name in BUDGET) + int(any(BLURRED.get(name, (0, 0))))


def warned(log, name, capped, lowered, pooled):
    """Failures unless `log` holds `capped` warnings of textures the cap shrinks, `lowered` of
    textures lowered to fit and `pooled` of a shadow pool too small."""
    with open(log) as f:
        lines = [line for line in f if line.startswith("pika: warning: ")]
    failures = []
    for kind, mark, expected in (("textures shrunk by the cap", "by pika:textureCap", capped),
                                 ("textures shrunk by the budget", "to fit the device's memory budget", lowered),
                                 ("a shadow pool too small", "and pika:shadowPool holds", pooled)):
        count = sum(mark in line for line in lines)
        if count != expected:
            failures.append(f"{name}: {count} warnings of {kind}, not {expected}")
    return failures


def check(name, renders, references, aovs):
    """Compares a host's frames with the references and the controls."""
    failures = []
    changed = set()
    times = sorted(renders)
    for t in times:
        problems = compare(renders[t], references[t][None], aovs)
        print(f"  frame {t}: {'; '.join(problems) or 'matches'}")
        failures += [f"{name} frame {t}: {p}" for p in problems]
    t = times[-1]
    for control in CONTROLS:
        frame = times[0] if control == "frame 1's reference" else t
        perturbed = references[frame][None if frame != t else control]
        if not compare(perturbed, references[t][None], aovs):
            continue
        changed.add(control)
        caught = bool(compare(renders[t], perturbed, aovs, references[t][None]))
        print(f"  control '{control}': {'fails, as it must' if caught else 'PASSES'}")
        if not caught:
            failures.append(f"{name}: the control '{control}' passes")
    return failures, changed


def main():
    out = sys.argv[1]
    record = sys.argv[2:] == ["usdrecord"]
    os.makedirs(out, exist_ok=True)
    scenes = sorted(os.path.join(SCENES, s) for s in os.listdir(SCENES) if s not in LAYERS)
    stages = {s: Usd.Stage.Open(s) for s in scenes}
    sizes = {s: tuple(UsdRender.Settings(stages[s].GetPrimAtPath(SETTINGS)).GetResolutionAttr().Get())
             for s in scenes}
    with (concurrent.futures.ProcessPoolExecutor(REFERENCES) as compute,
          concurrent.futures.ThreadPoolExecutor(RENDERS) as render):
        # The pool forks at its first job, which must be before the render threads start.
        compute.submit(len, scenes).result()
        husked = scenes if shutil.which("husk") else []
        refs = {s: compute.submit(references, s, *sizes[s], HUSK) for s in husked}
        husks = {s: render.submit(husk, s, out, times(stages[s])) for s in husked}
        again = os.path.join(out, "again")
        os.makedirs(again, exist_ok=True)
        agains = {s: render.submit(husk, s, again, times(stages[s])) for s in husked
                  if os.path.basename(s)[:-5] in REPEATED}
        records = {s: render.submit(usdrecord, s, out, sizes[s][0], times(stages[s])) for s in scenes} if record else {}
        recorded = {}
        for s, job in records.items():
            # usdrecord takes the image's height from the camera's aspect.
            shape = next(iter(job.result()[0].values()))["color"].shape
            host = usdrecord_host()
            if (s not in refs or shape[:2] != sizes[s][::-1] or differs(stages[s], host)
                    or os.path.basename(s)[:-5] in CAMERA_LIT):
                recorded[s] = compute.submit(references, s, shape[1], shape[0], host)
        failures = []
        changed = set()
        for s in scenes:
            name = os.path.splitext(os.path.basename(s))[0]
            if s in husks:
                renders, log = husks[s].result()
                print(f"{name} in husk ({log}):")
                f, c = check(name, renders, refs[s].result(), ["color", "depth", "normal"])
                failures += f + warned(log, name, CAPPED.get(name, 0), budget_warnings(name), POOLED.get(name, 0))
                frames = stats(log.replace(".husk.log", ".husk.stats"))
                failures += blurred(frames, name) + cached(frames, name)
                if s in agains:
                    failures += repeated(name, renders, agains[s].result()[0])
                changed |= c
            if s not in records:
                continue
            renders, log = records[s].result()
            print(f"{name} in usdrecord ({log}):")
            failures += check(f"{name} (usdrecord)", renders, (recorded.get(s) or refs[s]).result(), ["color"])[0]
            failures += warned(log, f"{name} (usdrecord)", 0, budget_warnings(name), 0)
    # Some controls change only depth or normals, which only husk's images hold.
    for control in CONTROLS if husked else []:
        if control not in changed:
            failures.append(f"the control '{control}' changes no scene's reference")
    if not husked and not records:
        failures.append("no host rendered: husk is not on PATH and usdrecord was not asked for")
    if not husked:
        print("husk is not on PATH: depth, normals and husk's stats were not checked")
    for f in failures:
        print(f"FAIL: {f}")
    print("PASS" if not failures else f"{len(failures)} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
