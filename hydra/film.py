"""Checks Pika's film while frames change, through UsdImagingGL as
hydra/engine.py drives it: that history follows a panning camera, that a
light's change shows through it, and that a converged still is the same
whatever came before it.

    hython hydra/film.py <panned.usda>

Needs a display (check.sh runs it on its Xvnc) and the delegate on
PXR_PLUGINPATH_NAME. The scene's bars and their shadows are thinner than a
pixel, so a frame alone aliases. Temporal history has no closed form
(design.md), so each frame is compared with the converged still at its
camera, drawn in a session of its own, by the mean of their 8-bit channel
differences. A frame's bound is the error of a frame drawn alone there, a
new session's first, which has no history:

- Panning by STEP pixels a frame, each frame once history has built up must
  be within its bound. The control is the still a frame behind, what
  ghosting shows, which must not be.
- Halving the sun's intensity with the camera held, each frame once LIT
  have passed must be within the bound at the new light. The control is the
  old still, which must not be.
- Converging at one camera reached from two others must give the same
  image bit for bit, as the image the view stopped at fades out of the
  samples by the last. The control is the images part way, which must
  differ: the image stopped at counts there, so the check is not vacuous.
"""

import sys

from pxr import Gf, Usd, UsdLux

# Importing the check would leave a __pycache__ in the checkout.
sys.dont_write_bytecode = True
from engine import SAMPLES, SIZE, accumulate, framing, gl_context, pixels, scene  # noqa: E402

# The camera's step a frame, in pixels at the scene's centre: not a whole number, so the bars
# fall across pixels differently each frame.
STEP = 1.35
FRAMES = 32
# Frames before history is judged: its blend keeps about 20.
WARM = 12
# Frames the light's change may take to show.
LIT = 4
# The still sample at which the images part way are compared.
PART_WAY = 7


def error(a, b):
    """The mean 8-bit channel difference between two images."""
    total = sum(abs((p >> s & 255) - (q >> s & 255)) for p, q in zip(a, b) for s in (0, 8, 16))
    return total / (3 * len(a))


def camera(stage, right):
    """The framing camera's view and projection, moved `right` pixels at the scene's centre."""
    frustum = framing(stage)
    view = frustum.ComputeViewMatrix()
    projection = frustum.ComputeProjectionMatrix()
    depth = -view.Transform(frustum.ComputeLookAtPoint())[2]
    # A pixel is 2 / SIZE in normalized device x, which is P[0][0] x / -z.
    shift = right * 2 / SIZE * depth / projection[0][0]
    return view * Gf.Matrix4d().SetTranslate(Gf.Vec3d(-shift, 0, 0)), projection


def converged(engine, stage, params, fbo, state):
    engine.SetCameraState(*state)
    while True:
        engine.Render(stage.GetPseudoRoot(), params)
        if engine.IsConverged():
            return pixels(fbo)


def alone(path, fbo, state, edit=None):
    """A frame drawn with no history: a new session's first, after `edit(stage)`."""
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


def halve(stage):
    """Halves the sun in the stage's session layer: stages opened from one file share its
    layer, which would edit every session at once."""
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


def main(panned):
    keep = gl_context()  # noqa: F841
    fbo = keep[3]
    pan(panned, fbo)
    light(panned, fbo)
    path_independence(panned, fbo)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(*sys.argv[1:])
