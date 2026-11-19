"""Checks Pika's film while frames change, through UsdImagingGL. Each frame is compared with the
converged still at its camera, and must be nearer to it than a frame drawn alone.

    python3 hydra/film.py <panned.usda> <penumbra.usda>

Needs a display and the delegate on PXR_PLUGINPATH_NAME.
"""

import sys

from pxr import Gf, Usd, UsdGeom, UsdLux

# Importing the check would leave a __pycache__ in the checkout.
sys.dont_write_bytecode = True
from engine import SAMPLES, SIZE, accumulate, framing, gl_context, pixels, scene  # noqa: E402

# The camera's step a frame, in pixels at the scene's centre.
STEP = 1.35
FRAMES = 32
# Frames before history is judged.
WARM = 12
# Frames the light's change may take to show.
LIT = 4
# The still sample at which the images part way are compared.
PART_WAY = 7
# The still samples a still camera's history must match in a penumbra, and the fewer that must not.
HELD = 16
FEWER = 4


def error(a, b, mask=None):
    """The mean 8-bit channel difference between two images, over the pixels `mask` holds."""
    mask = mask or [True] * len(a)
    total = sum(abs((p >> s & 255) - (q >> s & 255)) for p, q, m in zip(a, b, mask) if m for s in (0, 8, 16))
    return total / (3 * sum(mask))


def edges(image):
    """Per pixel, whether any of its 3×3 neighbours differs from it."""
    def at(x, y):
        return image[min(max(y, 0), SIZE - 1) * SIZE + min(max(x, 0), SIZE - 1)]
    return [any(at(x + i, y + j) != at(x, y) for i in (-1, 0, 1) for j in (-1, 0, 1)) for y in range(SIZE) for x in range(SIZE)]


def shift(frustum, right):
    """The view-space x that moves the scene's centre `right` pixels across the frustum's image."""
    depth = -frustum.ComputeViewMatrix().Transform(frustum.ComputeLookAtPoint())[2]
    # A pixel is 2 / SIZE in normalized device x, which is P[0][0] x / -z.
    return right * 2 / SIZE * depth / frustum.ComputeProjectionMatrix()[0][0]


def camera(stage, right):
    """The framing camera's view and projection, moved `right` pixels at the scene's centre."""
    frustum = framing(stage)
    moved = Gf.Matrix4d().SetTranslate(Gf.Vec3d(-shift(frustum, right), 0, 0))
    return frustum.ComputeViewMatrix() * moved, frustum.ComputeProjectionMatrix()


def offset(stage, right):
    """The world offset that moves the scene's centre `right` pixels across the framing camera."""
    frustum = framing(stage)
    return frustum.ComputeViewMatrix().GetInverse().TransformDir(Gf.Vec3d(shift(frustum, right), 0, 0))


def place(stage, offset):
    """Moves the bars and the floor in the stage's session layer, as `halve` edits the sun."""
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path in ("/world/bars", "/world/floor"):
            mesh = UsdGeom.Xformable(stage.GetPrimAtPath(path))
            ops = mesh.GetOrderedXformOps()
            (ops[0] if ops else mesh.AddTranslateOp()).Set(offset)


def bend(stage, offset):
    """Moves the bars and the floor by their points, their transforms still, and has the bars
    refined."""
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Mesh(stage.GetPrimAtPath("/world/bars")).GetSubdivisionSchemeAttr().Set("bilinear")
        for path in ("/world/bars", "/world/floor"):
            authored = stage.GetRootLayer().GetAttributeAtPath(path + ".points").default
            points = [p + Gf.Vec3f(offset) for p in authored]
            UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetPointsAttr().Set(points)


def unshadow(stage):
    """Turns the sun's shadow off in the stage's session layer."""
    sun = stage.GetPrimAtPath("/world/sun")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdLux.ShadowAPI.Apply(sun).CreateShadowEnableAttr(False)


def converged(engine, stage, params, fbo, state):
    engine.SetCameraState(*state)
    while True:
        engine.Render(stage.GetPseudoRoot(), params)
        if engine.IsConverged():
            return pixels(fbo)


def alone(path, fbo, state, edit=None):
    """A frame drawn with no history."""
    engine, stage, params = scene(path)
    if edit:
        edit(stage)
    engine.SetCameraState(*state)
    engine.Render(stage.GetPseudoRoot(), params)
    return pixels(fbo)


def pan(path, fbo):
    engine, stage, params = scene(path)
    reference, reference_stage, _ = scene(path)
    states = [camera(stage, STEP * (t - FRAMES / 2)) for t in range(FRAMES)]
    stills = [converged(reference, reference_stage, params, fbo, s) for s in states]
    converged(engine, stage, params, fbo, states[0])
    shown = {}
    for t in range(1, FRAMES):
        engine.SetCameraState(*states[t])
        engine.Render(stage.GetPseudoRoot(), params)
        # A render shows the frame the one before drew.
        if t >= 2:
            shown[t - 1] = pixels(fbo)
    for t in range(WARM, FRAMES - 1):
        bound = error(alone(path, fbo, states[t]), stills[t])
        drawn, behind = error(shown[t], stills[t]), error(stills[t - 1], stills[t])
        if not drawn < bound:
            sys.exit(f"film: panning, frame {t} is {drawn:.2f} from its still, not within a frame alone's {bound:.2f}")
        if behind < bound:
            sys.exit(f"film: the control failed: the still a frame behind is {behind:.2f} from frame {t}'s, within {bound:.2f}")
        print(f"film: panning, frame {t} is {drawn:.2f} from its still, a frame alone {bound:.2f}, the still a frame behind {behind:.2f}")


def slide(path, fbo, move, sliding):
    """`move` moves the bars and the floor (`place` or `bend`)."""
    engine, stage, params = scene(path)
    reference, reference_stage, _ = scene(path)
    unshadow(stage)
    unshadow(reference_stage)
    state = camera(stage, 0)
    places = [offset(stage, STEP * (t - FRAMES / 2)) for t in range(FRAMES)]
    stills = []
    for p in places:
        move(reference_stage, p)
        stills.append(converged(reference, reference_stage, params, fbo, state))
    move(stage, places[0])
    converged(engine, stage, params, fbo, state)
    shown = {}
    for t in range(1, FRAMES):
        move(stage, places[t])
        engine.Render(stage.GetPseudoRoot(), params)
        if t >= 2:
            shown[t - 1] = pixels(fbo)
    for t in range(WARM, FRAMES - 1):

        def moved(s):
            unshadow(s)
            move(s, places[t])

        bound = error(alone(path, fbo, state, moved), stills[t])
        drawn, behind = error(shown[t], stills[t]), error(stills[t - 1], stills[t])
        if not drawn < bound:
            sys.exit(f"film: {sliding}, frame {t} is {drawn:.2f} from its still, not within a frame alone's {bound:.2f}")
        if behind < bound:
            sys.exit(f"film: the control failed: the still a frame behind is {behind:.2f} from frame {t}'s, within {bound:.2f}")
        print(f"film: {sliding}, frame {t} is {drawn:.2f} from its still, a frame alone {bound:.2f}, the still a frame behind {behind:.2f}")


def tick(stage, t):
    """Moves a quad hidden under the floor in the stage's session layer."""
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        quad = UsdGeom.Mesh.Define(stage, "/world/tick")
        if not quad.GetOrderedXformOps():
            quad.CreatePointsAttr([(-0.1, -0.1, 0), (0.1, -0.1, 0), (0.1, 0.1, 0), (-0.1, 0.1, 0)])
            quad.CreateFaceVertexCountsAttr([4])
            quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            quad.CreateSubdivisionSchemeAttr("none")
            quad.AddTranslateOp()
        quad.GetOrderedXformOps()[0].Set(Gf.Vec3d(0.001 * (t % 7), 0, -3))


def rotate(path, fbo):
    engine, stage, params = scene(path)
    stills = {s: image for image, s in accumulate(engine, stage, params, fbo)}
    final = stills[SAMPLES - 1]
    engine, stage, params = scene(path)
    unshadow(stage)
    lit = accumulate(engine, stage, params, fbo)[-1][0]
    # What the shadow changes, less the edges a shadow's change would mix in.
    penumbrae = [a != b and not e for a, b, e in zip(final, lit, edges(lit))]
    bound, fewer = (error(stills[n - 1], final, penumbrae) for n in (HELD, FEWER))
    if fewer < bound:
        sys.exit(f"film: the control failed: {FEWER} still samples are {fewer:.2f} from the converged still, within {HELD}'s {bound:.2f}")
    engine, stage, params = scene(path)
    tick(stage, 0)
    accumulate(engine, stage, params, fbo)
    drawn = []
    for t in range(1, FRAMES):
        tick(stage, t)
        engine.Render(stage.GetPseudoRoot(), params)
        if t > WARM:
            drawn.append(error(pixels(fbo), final, penumbrae))
    mean = sum(drawn) / len(drawn)
    if not mean < bound:
        sys.exit(f"film: playing with the camera still, the penumbrae are {mean:.2f} from the converged still, not within {HELD} still samples' {bound:.2f}")
    print(f"film: playing with the camera still, the penumbrae are {mean:.2f} from the converged still, within {HELD} still samples' {bound:.2f}; {FEWER}'s are {fewer:.2f}")


def halve(stage):
    """Halves the sun in the stage's session layer."""
    sun = UsdLux.DistantLight(stage.GetPrimAtPath("/world/sun"))
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        sun.GetIntensityAttr().Set(sun.GetIntensityAttr().Get() / 2)


def light(path, fbo):
    reference, reference_stage, params = scene(path)
    state = camera(reference_stage, 0)
    halve(reference_stage)
    after = converged(reference, reference_stage, params, fbo, state)
    bound = error(alone(path, fbo, state, halve), after)
    engine, stage, params = scene(path)
    before = converged(engine, stage, params, fbo, state)
    halve(stage)
    old = error(before, after)
    if old < bound:
        sys.exit(f"film: the control failed: the old light's still is {old:.2f} from the new one's, within {bound:.2f}")
    for k in range(1, LIT + 4):
        engine.Render(stage.GetPseudoRoot(), params)
        drawn = error(pixels(fbo), after)
        if k > LIT and not drawn < bound:
            sys.exit(f"film: {k} frames after the light halved, the image is {drawn:.2f} from the new still, not within {bound:.2f}")
    print(f"film: from {LIT + 1} frames after the light halved, each image is within a frame alone's {bound:.2f} of the new still, the last {drawn:.2f}; the old still is {old:.2f} from it")


def path_independence(path, fbo):
    finals, parts = [], []
    for start in (-8, 8):
        engine, stage, params = scene(path)
        converged(engine, stage, params, fbo, camera(stage, start))
        engine.SetCameraState(*camera(stage, 0))
        frames = accumulate(engine, stage, params, fbo)
        images = {s: image for image, s in frames}
        finals.append(images[SAMPLES - 1])
        parts.append(images[PART_WAY])
    if finals[0] != finals[1]:
        differ = sum(a != b for a, b in zip(*finals))
        sys.exit(f"film: converged from either side, {differ} pixels differ")
    differ = sum(a != b for a, b in zip(*parts))
    if differ == 0:
        sys.exit(f"film: the control failed: at sample {PART_WAY} the images from either side are the same")
    print(f"film: converged from either side the same, {differ} pixels apart at sample {PART_WAY}")


def main(panned, penumbra):
    keep = gl_context()  # noqa: F841
    fbo = keep[3]
    pan(panned, fbo)
    slide(panned, fbo, place, "sliding")
    slide(panned, fbo, bend, "sliding by points")
    rotate(penumbra, fbo)
    light(panned, fbo)
    path_independence(panned, fbo)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
