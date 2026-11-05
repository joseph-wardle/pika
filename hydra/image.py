"""Pika's image check: renders each scene in hydra/images/ in husk, for
frames 1 and 2, and compares its color, depth and normal AOVs with a ray cast
from USD, one ray through each pixel centre. The reference reads the stage,
never Pika's store, so the two cannot share a mistake.

    hython hydra/image.py <out dir> [usdrecord]

Run it from a shell set up as hydra/check.sh sets one up. With `usdrecord`,
it also renders each scene's color through usdrecord (usdview's engine),
which needs an X display in DISPLAY. The renders run RENDERS at a time and
the references on every core; the results print in the scenes' order.

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
- Color is the material's albedo under the headlight plus its emission,
  with alpha 1: the albedo times the normal against the direction toward
  the eye, clamped, the normal turned toward the eye first as MaterialX's
  BSDFs turn it, under a headlight of π along the camera's axis, as
  src/surface.rs sets it, so a facing surface shows its albedo. A mesh
  with no material bound is clay: albedo 18 % grey, no emission. An
  OpenPBR surface (hydra/images/pbr.usda) has albedo base_weight ×
  base_color and emission emission_luminance × emission_color, a Standard
  Surface base × base_color and emission × emission_color, unauthored
  inputs at the defaults the host's libraries give them; their specular
  weight must be 0 and no other input authored or connected, since the
  rest has no closed form. A UsdPreviewSurface (hydra/images/wrapped.usda)
  has albedo diffuseColor and emission emissiveColor, a value or the rgb a
  UsdUVTexture samples, under the specular workflow with a black
  specularColor and nothing else authored; MaterialX layers a Schlick lobe
  with F90 white over its diffuse, so it is checked within 30° of facing.
  An OpenPBR surface is checked only within 35°
  of facing the eye: at specular weight 0 its dielectric layer has an IOR
  of 1, but MaterialX still attenuates the base by the layer's directional
  albedo, an analytic fit with F90 = 1 whatever the IOR, which takes 0.2 %
  of the base at facing (within the colour tolerance for a base under 1),
  nothing near 25° and 13 % at 67°, so a pixel seen from further round is
  refused. Either surface's normal (Standard Surface's normal, OpenPBR's
  geometry_normal) may be a normalmap node of an image node's vector3
  sample at a primvar (hydra/images/normalmap.usda), which perturbs the
  normal it is lit with: the sample, each channel × 2 − 1 (a zero sample
  standing for (0, 0, 1)), taken as x along the tangent, y along the
  bitangent and z along the normal, normalized, the frame interpolated
  across the face from its vertices' and each direction normalized first.
  A vertex's tangent is Lengyel's, as shaders/tangent.slang computes it:
  the directions its mesh's st u and v increase in, solved from each of
  its drawn triangles' edges and st deltas in the mesh's own space and
  summed over them, a vertex being a unique point, normal and st, as Pika
  welds them, a faceVarying st by value, so an edge whose faces agree on
  it sums both faces' triangles and a seam keeps them apart
  (hydra/images/bent.usda); the tangent is the u direction made orthogonal to the
  vertex's normal, and the bitangent the normal's cross with it, negated
  where the mapping is mirrored (the summed v direction is against the
  cross). Both move with the mesh's transform, not its normals' inverse
  transpose. A triangle without st area adds nothing, and a vertex whose
  u directions cancel, or without st, gets the y axis crossed with its
  normal (the x axis when the normal is near y), normalized, and
  handedness 1. The map is read as Pika bakes a vector3 image, at 8 bits,
  so the fixture's values are multiples of 1/255. A mesh
  bound to a MaterialX surface whose
  emission is a uniform EDF of an image sampled at a primvar
  (hydra/images/textured.usda) emits the image's sample at the hit's
  primvar, the one the material's reader names (hydra/images/primvar.usda)
  or st when the image's texcoord is unconnected or a texcoord node of
  index 0: UV0, the texcoord MaterialX's nodes read by default, which
  Storm and Karma read as st (hydra/images/uv0.usda),
  under the image node's address modes, or a UsdUVTexture's wrapS and
  wrapT, read at the frame (hydra/images/rewrapped.usda), whose
  useMetadata, USD's default when none is authored, is what the file's
  wrapmodes metadata states, black where it states none, as Storm reads it
  (hydra/images/wrapped.usda); with st (0, 0) at the image's
  bottom-left corner as USD and MaterialX read it, the channels padded
  with 0 to three. The sample is bilinear within a level, at the level the
  hit's footprint picks as a Vulkan sampler with linear mip filtering and
  no anisotropy picks it (hydra/images/mips.usda): the footprint is the st
  step from the pixel's centre to the next pixel's, right and down, on the
  hit's plane; the longer step, in texels of level 0, is 2^λ, and levels
  ⌊λ⌋ and ⌈λ⌉ are blended by λ's fraction, level 0 alone where the image is
  magnified. The levels are the file's stored ones from the first within
  the texture cap, the render setting pika:textureCap, 1024 unless the
  render settings prim authors it (hydra/images/capped.usda); the
  reference refuses a level the file lacks and a cap no stored level fits.
  Pika warns, once per texture it loads, of a source the cap shrank. husk
  reads the setting off the prim; usdrecord passes no render settings, so
  it draws under the default, and the reference knows. The step is the
  GPU's own on a surface facing the camera, where st is affine on screen;
  no scene minifies any other kind.
  A file naming <UDIM> is a set of
  tiles (hydra/images/udim.usda): the hit samples tile 1001 + ⌊u⌋ + 10 ⌊v⌋
  at (u − ⌊u⌋, v − ⌊v⌋), under the same modes within the tile, 0 where the
  set has no such file, as USD numbers tiles and Storm and Karma read
  them. A color3 image's sample
  (hydra/images/colorspace.usda) is emitted in the working space, linear
  Rec.709, from the colour space its file input declares: srgb_texture
  decoded by the sRGB curve, acescg by the matrix from AP1 through CIE
  XYZ, with a Bradford adaptation from the ACES white to D65, to Rec.709,
  as the ACES configs define it, raw as read, and none (auto, USD's
  default) as sRGB for an 8-bit source and as read for a float one, the
  texels converted before they are filtered. A vector2, vector3 or float
  image declaring none is read as it is whatever its depth, as Karma reads
  it: the auto rule is for colour. Other spaces are refused. On a subdivided mesh the primvar is
  refined with the points (hydra/images/subdivided.usda): a vertex primvar
  by the points' rules, a faceVarying one by the same rules over its own
  values, welded by point and value first, so it is smooth across an edge
  whose faces agree on it and creased along a seam, as USD's default
  faceVaryingLinearInterpolation, cornersPlus1, reads it; a value of a
  single face stays put. That is the reading of Houdini's subdivision,
  Karma and Blender; Storm keeps every corner apart, which interpolates
  each cage face linearly. Seams ending inside the surface or meeting
  another seam or a boundary at a vertex are refused. A mesh bound to a material whose surface
  shader names a MaterialX nodedef (ND_...) the host's libraries do not
  define emits pink, (1, 0, 1), as Pika draws a material it cannot build,
  when the host hands the network over: husk and UsdImaging's scene-index
  path do, UsdImaging's legacy path drops a network whose surface shader
  the shader registry does not know, so Pika sees no surface and draws
  clay. usdrecord takes the path USDIMAGINGGL_ENGINE_ENABLE_SCENE_INDEX
  names, off by default in Houdini 21 and on in 22. Other materials are
  refused, not approximated.
- Pixels with no hit keep the clear values: color 0, depth 1, normal 0.

Pixels next to a change of face (silhouettes, face boundaries) are compared
only for coverage, since a rasterizer and a ray decide them differently. Away
from them, values must agree within TOLERANCE, colour also allowed 1/256 of
its value: the GPU decodes sRGB texels to about nine significant bits (RTX
4070 Ti SUPER, driver 580), so a sample of a light grey is off by up to
half of 1/256 before it is filtered.

Each control perturbs the reference. A control must fail on every scene
and host whose reference it changes, and must change at least one scene's.
"""

import collections
import concurrent.futures
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
# The delegate's default texture cap (hydra/plugin.cpp).
CAP = 1024
# The scenes whose texture the cap shrinks, which husk must warn of once.
CAPPED = {"capped": 1}
FRAMES = (1, 2)
# Each render is a Houdini process of about 1 GB with a few busy threads.
RENDERS = 8
# Color allows for usdrecord's half floats.
TOLERANCE = {"color": 2e-3, "depth": 2e-6, "normal": 2e-4}
CONTROLS = ["focal length +0.5 %", "half a pixel right", "aperture cropped", "unconverted depth",
            "unflipped normals", "flat normals", "instancer transform dropped", "outer instances dropped",
            "orientations misread", "authored normals on subdivided meshes", "holes drawn",
            "face normals averaged", "left-handed corners unswapped", "frame 1's smooth normals",
            "cage drawn on subdivided meshes", "boundary rules of the interior",
            "purposes ignored", "culled by box centre", "headlight from the eye",
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
            "the cap ignored", "frame 1's reference"]
# The controls under which a subdivided mesh is drawn as its cage.
CAGE_DRAWN = ("cage drawn on subdivided meshes", "authored normals on subdivided meshes")
# ACEScg to linear Rec.709 (see the module's doc).
ACESCG_TO_REC709 = np.array([[1.705051, -0.621792, -0.083259],
                             [-0.130256, 1.140805, -0.010548],
                             [-0.024003, -0.128969, 1.152972]])
# A material's closed form: its albedo under the headlight, its emission, the
# image (file, address modes, primvar, colour space) whose sample it emits, if any, the
# image whose sample perturbs its normal, if any, and the angle from facing
# the eye within which the form holds (None: all).
Look = collections.namedtuple("Look", "albedo emission image normal facing")
CLAY = Look((0.18,) * 3, (0.0,) * 3, None, None, None)
PINK = Look((0.0,) * 3, (1.0, 0.0, 1.0), None, None, None)
# Each lit surface's inputs the reference reads: the albedo's weight and
# colour, the emission's weight and colour, the specular weight, the
# normal, and the facing angle. OpenPBR's specular weight only modulates the IOR, and
# MaterialX's dielectric layer attenuates its base by a directional albedo
# fitted with F90 = 1 at any IOR, so grazing angles darken at weight 0 (see
# the module's doc); Standard Surface's weight is the layer's own, which 0
# switches off.
LIT = {
    "ND_open_pbr_surface_surfaceshader": ("base_weight", "base_color", "emission_luminance",
                                          "emission_color", "specular_weight", "geometry_normal", 35),
    "ND_standard_surface_surfaceshader": ("base", "base_color", "emission", "emission_color",
                                          "specular", "normal", None),
}
# What a host hands Pika: the purposes it draws, whether a material whose
# surface shader the shader registry does not know arrives with its network,
# and whether the render settings prim's Pika settings arrive (see the
# module's doc).
Host = collections.namedtuple("Host", "purposes delivers_unknown settings")
HUSK = Host(("default", "render"), True, True)


def usdrecord_host():
    scene_index = Tf.GetEnvSetting("USDIMAGINGGL_ENGINE_ENABLE_SCENE_INDEX")
    return Host(("default", "proxy"), scene_index, False)


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
    counts, face vertex indices and holes, and each refined face's cage
    face."""
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
            # OpenSubdiv rotates a quad's children to keep its parametric
            # orientation, which decides the diagonal each child is split on.
            rotation = j if len(face) == 4 else 0
            out_indices += child[-rotation:] + child[:-rotation] if rotation else child
    return refined, out_counts, out_indices, set(out_holes), out_parents


def surface(mesh, time, control=None):
    """The points, face vertex counts, face vertex indices and holes a
    mesh is drawn with at `time`, and each drawn face's cage face."""
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
    """Welds a faceVarying primvar's values by point and value, as Pika
    does: the distinct values, and the one each corner reads. They are the
    primvar's own topology: continuous across an edge whose faces agree on
    it, a seam where they do not. Raises on a seam ending inside the
    surface or meeting another seam or a boundary at a vertex, which the
    reference does not refine."""
    welded, index, corner_value = [], {}, []
    for point, value in zip(indices, values):
        key = point if control == "st seams smoothed over" else (point, tuple(value))
        if key not in index:
            index[key] = len(welded)
            welded.append(value)
        corner_value.append(index[key])
    # Each edge's sides: the value each of its faces reads at each end.
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
    """The primvar as the faces drawn read it: its values and their
    interpolation. On a subdivided mesh they are refined with the points,
    one per refined corner: a vertex primvar by the points' rules, a
    faceVarying one by the same rules over its welded values (see `weld`),
    under cornersPlus1, the only faceVaryingLinearInterpolation the
    reference knows."""
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
    # Storm's reading: it keeps every corner apart, so under cornersPlus1
    # every value is a corner and every edge a seam, which is linear.
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
    """The host's MaterialX libraries, as the bake tool loads them."""
    libraries = mx.createDocument()
    search = mx.FileSearchPath(os.path.join(os.environ["HFS"], "houdini", "materialx"))
    mx.loadLibraries(["libraries"], search, libraries)
    return libraries


NODEDEFS = nodedefs()


def unknown(shader):
    """Whether the shader names a nodedef the libraries do not define."""
    return shader.GetShaderId().startswith("ND_") and not NODEDEFS.getNodeDef(shader.GetShaderId())


def uv0(image):
    """Whether the image node samples UV0, the texcoord MaterialX's nodes
    read by default: its texcoord input unconnected, or a texcoord node of
    index 0."""
    if not image.GetInput("texcoord"):
        return True
    node = upstream(image, "texcoord")
    if not node or node.GetShaderId() != "ND_texcoord_vector2":
        return False
    index = node.GetInput("index")
    return not index or index.Get() == 0


def sampled(image, kind, time, control=None):
    """The file, address modes (u, v) at `time`, primvar and declared colour
    space (auto when none for a colour, raw for the rest) of an
    ND_image_<kind> node sampled at UV0 (st) or a geompropvalue reader, or
    of a UsdUVTexture (`kind` rgb, its output) sampled at st or a
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
    # USD resolves a path to the file it names, which a <UDIM> pattern is
    # not; the pattern is anchored to its layer, as UsdImaging anchors it.
    path = file.Get().resolvedPath
    if "<UDIM>" in file.Get().path:
        path = file.GetAttr().GetPropertyStack(Usd.TimeCode.Default())[0].layer.ComputeAbsolutePath(file.Get().path)
    if usd:
        modes = tuple(wrap(image, name, path, time, control) for name in ("wrapS", "wrapT"))
    else:
        modes = tuple(image.GetInput(m).Get(time) if image.GetInput(m) else "periodic"
                      for m in ("uaddressmode", "vaddressmode"))
    colour = usd or kind == "color3" or control == "8-bit data read as sRGB"
    space = file.GetAttr().GetColorSpace() or ("auto" if colour else "raw")
    return (path, modes, primvar, space)


def wrap(image, name, path, time, control=None):
    """A UsdUVTexture's wrapS or wrapT at `time` in the image nodedefs'
    words: repeat is periodic, and useMetadata, USD's default when none is
    authored, is what the file's metadata states for that axis."""
    token = image.GetInput(name).Get(time) if image.GetInput(name) else "useMetadata"
    if token != "useMetadata":
        return "periodic" if token == "repeat" else token
    if control == "useMetadata read as periodic":
        return "periodic"
    return metadata(path, control)[name == "wrapT"]


def metadata(path, control=None):
    """The wrap modes (u, v) the file's wrapmodes attribute states, as
    maketx writes it ("black,clamp", or one word for both), black for an
    axis it states none or another for, as Storm reads it; a set's are its
    first tile's, as the bake tool reads them."""
    if "<UDIM>" in path:
        path = next(p for p in (path.replace("<UDIM>", str(n)) for n in range(1001, 1101)) if os.path.isfile(p))
    file = oiio.ImageInput.open(path)
    stated = "" if control == "file wrap metadata ignored" else file.spec().getattribute("wrapmodes") or ""
    file.close()
    words = stated.split(",")
    return tuple(w if w in ("black", "clamp", "periodic", "mirror") else "black" for w in (words + words)[:2])


def lit(surface, time, control=None):
    """The Look of an OpenPBR or Standard Surface shader with its specular
    off (hydra/images/pbr.usda), its inputs read at `time`, its normal
    mapped when connected (hydra/images/normalmap.usda)."""
    *names, bump, facing = LIT[surface.GetShaderId()]
    path = surface.GetPath()
    authored = {i.GetBaseName(): i for i in surface.GetInputs()}
    extra = sorted(set(authored) - set(names) - {bump})
    connected = [n for n, i in authored.items() if i.HasConnectedSource() and n != bump]
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
        if name in authored:
            v = authored[name].Get(time)
        else:
            v = NODEDEFS.getNodeDef(surface.GetShaderId()).getActiveInput(name).getValue()
        return tuple(v) if hasattr(v, "__len__") else v

    if value(names[4]) != 0:
        raise ValueError(f"{path}: the reference has no closed form for a specular lobe; set {names[4]} to 0")
    weight = 1.0 if control == "base weight ignored" else value(names[0])
    glow = 0.0 if control == "emission ignored" else value(names[2])
    return Look(tuple(weight * c for c in value(names[1])), tuple(glow * c for c in value(names[3])),
                None, normal, facing)


def preview(surface, time, control=None):
    """The Look of a UsdPreviewSurface (hydra/images/wrapped.usda): albedo
    diffuseColor, emission emissiveColor, a value or the rgb a UsdUVTexture
    samples, under the specular workflow with a black specularColor, since
    the rest has no closed form. MaterialX layers a Schlick lobe with F90
    white over the diffuse, which only grazing angles show, so the form
    holds within 30° of facing."""
    path = surface.GetPath()
    authored = {i.GetBaseName(): i for i in surface.GetInputs()}
    extra = sorted(set(authored) - {"diffuseColor", "emissiveColor", "useSpecularWorkflow", "specularColor"})
    connected = [n for n, i in authored.items() if i.HasConnectedSource() and n != "emissiveColor"]
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
    image, emission = None, value("emissiveColor")
    if "emissiveColor" in authored and authored["emissiveColor"].HasConnectedSource():
        image = sampled(upstream(surface, "emissiveColor"), "rgb", time, control)
        if not image:
            raise ValueError(f"{path}: the reference emits a UsdUVTexture's rgb sampled at a primvar, "
                             "nothing else connected")
        emission = (0.0,) * 3
    return Look(value("diffuseColor"), emission, image, None, 30)


def look(prim, host, time, control=None):
    """The closed form of the mesh's material: CLAY with none bound; PINK
    for a surface shader naming a nodedef the libraries do not define, when
    `host` delivers it; a UsdPreviewSurface's; a lit surface's; or the image file, its address
    modes (u, v), the primvar it is sampled at and its colour space, for a
    MaterialX surface whose emission is a uniform EDF of an image node's
    vector2 or vector3 sample through converts (hydra/images/textured.usda,
    hydra/images/mips.usda) or of its color3 sample
    (hydra/images/colorspace.usda)."""
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
    """The texels in the working space, linear Rec.709, from the colour
    space the asset declares (see the module's doc)."""
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
    """A 2×2 box of the texels (rows, columns, channels), the last row or
    column repeated where a side is odd, as the bake tool builds a level."""
    h, w = texels.shape[:2]
    padded = texels[np.minimum(np.arange(h + h % 2), h - 1)][:, np.minimum(np.arange(w + w % 2), w - 1)]
    return padded.reshape((h + 1) // 2, 2, (w + 1) // 2, 2, -1).mean((1, 3))


def bilinear(texels, st, modes, control=None):
    """The bilinear sample of one level's texels, rows from the bottom, at
    each st (an (n, 2) array), under the address modes (u, v)."""
    h, w = texels.shape[:2]
    f = st * [w, h] - 0.5
    if control == "nearest texel":
        f = np.floor(st * [w, h])
    i0 = np.floor(f).astype(int)
    fx, fy = (f - i0)[:, :1], (f - i0)[:, 1:]

    def address(i, n, mode):
        """The texel column or row `i` reads, in `padded`, whose border of
        0 is what black addresses past the edge."""
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
    """The trilinear sample of the image at `path`, or of the tile of the
    UDIM set `path` names, at each st (an (n, 2) array), rows from the
    bottom as USD reads st, under the address modes (u, v), in the working
    space; the channels padded with 0 to three. `step`, (n, 2, 2), is the
    st from each sample's pixel to the next one right and down, which picks
    the level among the file's from the first within `cap` texels a side
    (see the module's doc)."""
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
        """Level m's texels, the file's level first + m, in the working
        space, rows from the bottom."""
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
    """Lengyel's sums over the drawn triangles around each vertex (see the
    module's doc): the summed u and v directions, by the vertex `key` of
    each corner, from the corners' `st`."""
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
    """A vertex's unit tangent and handedness from its normal and summed u
    and v directions."""
    n = normalize(np.array(normal, float))
    t = u - n * np.dot(n, u)
    if not t.any() or control == "tangent from the first axis":
        t = np.cross((0, 1, 0) if abs(n[1]) < 0.9 else (1, 0, 0), n)
    t = normalize(t) if t.any() else np.array((1.0, 0.0, 0.0))
    mirrored = control != "bitangent from the winding" and np.dot(np.cross(n, t), v) < 0
    return t, -1.0 if mirrored else 1.0


def parts(prim, parents, host, time, control=None):
    """The Look of each drawn face: its materialBind GeomSubset's, the first
    of two claiming it, or the mesh's own for a face in none."""
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
    """Every drawn mesh instance's fan triangles, in world space: for each,
    the world points, the object normals at its corners, the normal matrix
    and the world matrix's linear part, whether it is left-handed, a face
    id, its face's Look, and, when a Look of the mesh samples an image,
    the sampled primvar at its corners and, for a normal map, their
    tangent frames."""
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
            at = FRAMES[0] if control == "frame 1's smooth normals" else time
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
    authored = stage.GetPrimAtPath(SETTINGS).GetAttribute("pika:textureCap")
    cap = authored.Get() if host.settings and authored else CAP
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
    around = footprint(to_clip, width, height, shift, world, nearest)
    hit = nearest >= 0
    i = nearest[hit]
    p, bary = p[hit], bary[hit]

    # Row vectors, as Gf multiplies them: v @ m.
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

    def sampled(field):
        """Each hit's sample of its look's `field` image at its st, 0 where
        it has none, and which hits have one."""
        out = np.zeros((len(p), 3))
        images = sorted({getattr(l, field) for l in looks if getattr(l, field)})
        which = np.array([images.index(getattr(l, field)) if getattr(l, field) else -1 for l in looks])[i]
        for k, (path, modes, _, space) in enumerate(images):
            on = which == k
            st = np.array([t["st"] or [[0, 0]] * 3 for t in tris])[i][on]
            at = np.einsum("nc,ncj->nj", bary[on], st)
            step = np.einsum("nkc,ncj->nkj", around[on], st) - at[:, None]
            out[on] = sample(path, at, step, modes, space, cap, control)
        return out, which >= 0

    # A normal map perturbs the normal before it is turned toward the eye.
    value, mapped = sampled("normal")
    if mapped.any() and control != "normal map ignored":
        value = np.where(dot(value, value)[:, None] == 0, (0.0, 0.0, 1.0), value * 2 - 1)
        frames = [t["frames"] or [((0, 0, 0), 1.0)] * 3 for t in tris]
        tangents = np.array([[list(t) for t, _ in f] for f in frames])[i]
        handed = np.array([[w for _, w in f] for f in frames])[i]
        # Each corner's bitangent, from its unnormalized object normal, as
        # the vertex stage computes it; the frame is interpolated, moved
        # with the mesh and normalized after.
        bitangents = handed[:, :, None] * np.cross(normals, tangents)
        r = np.array([[list(row) for row in t["linear"]] for t in tris])[i]
        t = normalize(np.einsum("ni,nij->nj", np.einsum("nc,ncj->nj", bary, tangents), r))
        b = normalize(np.einsum("ni,nij->nj", np.einsum("nc,ncj->nj", bary, bitangents), r))
        perturbed = normalize(t * value[:, :1] + b * value[:, 1:2] + shade * value[:, 2:])
        shade[mapped] = perturbed[mapped]
    # MaterialX's BSDFs turn a normal facing away from the eye toward it
    # before lighting.
    facing = shade
    facing[dot(facing, toward) < 0] *= -1
    light = toward if control == "headlight from the eye" else -np.array(f.ComputeViewDirection())
    lambert = np.clip(dot(facing, light), 0.0, 1.0)[:, None]
    limit = np.array([180 if l.facing is None else l.facing for l in looks])[i]
    if (dot(facing, toward) < np.cos(np.radians(limit))).any():
        raise ValueError("a surface is seen from further round than the angle its closed form holds within")
    albedo = np.array([l.albedo for l in looks])[i]
    emission = np.array([l.emission for l in looks])[i]
    color = np.c_[albedo * lambert + emission, np.ones(len(p))]
    color[:, :3] += sampled("image")[0]
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


def rays(unproject, width, height, x, y):
    """The ray through the image position (x, y), in pixels, rows down: its
    start on the near plane and the vector from there to the far plane."""
    ndc = np.c_[2 * x / width - 1, 1 - 2 * y / height]
    near = np.c_[ndc, -np.ones(len(x)), np.ones(len(x))] @ unproject
    far = np.c_[ndc, np.ones(len(x)), np.ones(len(x))] @ unproject
    o = near[:, :3] / near[:, 3:]
    return o, far[:, :3] / far[:, 3:] - o


def intersect(o, d, v0, e1, e2):
    """Moller-Trumbore: where the ray from `o` along `d` meets the plane of
    the triangle (v0, v0 + e1, v0 + e2): the point's barycentric u and v,
    its distance along the ray, and the determinant, 0 where the ray is
    parallel to the plane."""
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
    # A row at a time: every (pixel, triangle) pair.
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


def footprint(to_clip, width, height, shift, tris, nearest):
    """For each pixel `cast` hit, in image order: the barycentric
    coordinates, on its own triangle's plane, of the ray through the next
    pixel's centre to the right and of the one below, (hits, 2, 3). A
    texture's level follows from the st between a pixel and its neighbours,
    as the GPU differences its interpolated st across a pixel quad (see
    the module's doc)."""
    y, x = np.nonzero(nearest >= 0)
    i = nearest[y, x]
    unproject = np.linalg.inv(np.array(to_clip))
    out = np.zeros((len(x), 2, 3))
    for k, (dx, dy) in enumerate(((1, 0), (0, 1))):
        o, d = rays(unproject, width, height, x + 0.5 + shift + dx, y + 0.5 + dy)
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
        allowed = TOLERANCE[aov]
        if aov == "color":
            allowed = allowed + np.abs(ref[aov][..., :4]) / 256
        over = error > allowed
        over = over.reshape(over.shape[:2] + (-1,)).any(-1) & inner
        if over.any():
            slack = " plus 1/256 of the value" if aov == "color" else ""
            problems.append(f"{aov} differs by up to {error[over].max():.3g} (tolerance {TOLERANCE[aov]:g}{slack})")
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


def references(scene, width, height, host):
    """The reference and each control's, for each frame, of the scene at
    `scene`: a path, which a process pool can be handed."""
    stage = Usd.Stage.Open(scene)
    controls = [None] + [c for c in CONTROLS if c != "frame 1's reference"]
    return {t: {c: reference(stage, t, width, height, host, c) for c in controls} for t in FRAMES}


def differs(stage, host):
    """Whether `host`'s reference can differ from husk's at the same size:
    by a purpose authored, a Pika render setting it does not pass, or a
    material it does not deliver."""
    prims = list(stage.Traverse())
    purposed = any(UsdGeom.Imageable(p).GetPurposeAttr().HasAuthoredValue() for p in prims)
    settings = any(a.GetName().startswith("pika:") for a in stage.GetPrimAtPath(SETTINGS).GetAttributes())
    dropped = any(p.IsA(UsdShade.Shader) and unknown(UsdShade.Shader(p)) for p in prims)
    return purposed or (settings and not host.settings) or (dropped and host.delivers_unknown != HUSK.delivers_unknown)


def warned(log, name, expected):
    """A failure unless `log` holds `expected` texture warnings."""
    with open(log) as f:
        count = sum(line.startswith("pika: warning: texture ") for line in f)
    return [] if count == expected else [f"{name}: {count} texture warnings, not {expected}"]


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
    stages = {s: Usd.Stage.Open(s) for s in scenes}
    sizes = {s: tuple(UsdRender.Settings(stages[s].GetPrimAtPath(SETTINGS)).GetResolutionAttr().Get())
             for s in scenes}
    with (concurrent.futures.ProcessPoolExecutor() as compute,
          concurrent.futures.ThreadPoolExecutor(RENDERS) as render):
        # The pool forks before the render threads start.
        refs = {s: compute.submit(references, s, *sizes[s], HUSK) for s in scenes}
        husks = {s: render.submit(husk, s, out) for s in scenes}
        records = {s: render.submit(usdrecord, s, out, sizes[s][0]) for s in scenes} if record else {}
        recorded = {}
        for s, job in records.items():
            # usdrecord takes the image's height from the camera's aspect.
            shape = job.result()[0][FRAMES[0]]["color"].shape
            host = usdrecord_host()
            if shape[:2] != sizes[s][::-1] or differs(stages[s], host):
                recorded[s] = compute.submit(references, s, shape[1], shape[0], host)
        failures = []
        changed = set()
        for s in scenes:
            name = os.path.splitext(os.path.basename(s))[0]
            renders, log = husks[s].result()
            print(f"{name} in husk ({log}):")
            f, c = check(name, renders, refs[s].result(), ["color", "depth", "normal"])
            failures += f + warned(log, name, CAPPED.get(name, 0))
            changed |= c
            if s not in records:
                continue
            renders, log = records[s].result()
            print(f"{name} in usdrecord ({log}):")
            failures += check(f"{name} (usdrecord)", renders, recorded.get(s, refs[s]).result(), ["color"])[0]
            failures += warned(log, f"{name} (usdrecord)", 0)
    for control in CONTROLS:
        if control not in changed:
            failures.append(f"the control '{control}' changes no scene's reference")
    for f in failures:
        print(f"FAIL: {f}")
    print("PASS" if not failures else f"{len(failures)} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
