"""Renders two scenes through UsdImagingGL in this process, as usdview
does, and checks what the host then sees of Pika: the render stats
dictionary with the last frame's pairs and gpuMemoryUsed, which usdview's
HUD shows, and, for the second scene, whose geometry is the first's under a
material, that its first frame is not converged and shows the first scene's
clay while the material's build is out, that the converged frame does not,
and that a structural edit, which builds the material again, keeps the
converged look while its build is out, neither clay nor the new one,
whether the host edits the material in place (Houdini 22) or resyncs it,
removing it and inserting it again at its path (Houdini 21). Then, for a
third scene, a texture magnified, that changing the texture cap in one
session, from none to 4, 2 and none again, builds no material and draws
at each cap what a session started at it draws, that a raise whose
read fails, its baked file cut short, keeps the image it had and still
converges, and that once the file is whole again a cap change releases
it, drawing what a session at that cap draws.

    hython hydra/engine.py <clay.usda> <painted.usda> <mipped.usda>

Needs a display (check.sh runs it on its Xvnc) and the delegate on
PXR_PLUGINPATH_NAME. The controls are the dictionary before the first
frame, which has no frame pairs, the converged frame's pixel, which must
differ from the clay, the edit's converged pixel, which must differ
from the look it kept, and the fresh sessions' images at each cap, which
must differ from each other.
"""

import glob
import os
import sys
import time

from PySide6.QtCore import QSize
from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLFramebufferObject
from PySide6.QtWidgets import QApplication
from pxr import Gf, Usd, UsdGeom, UsdImagingGL, UsdShade

SIZE = 64
# How long a material's build may take before the check gives up.
PATIENCE = 60


def gl_context():
    """A current GL context with a bound framebuffer the engine's present
    task draws into, as a window's would be."""
    application = QApplication(sys.argv)
    surface = QOffscreenSurface()
    surface.setFormat(QSurfaceFormat())
    surface.create()
    context = QOpenGLContext()
    context.create()
    context.makeCurrent(surface)
    fbo = QOpenGLFramebufferObject(QSize(SIZE, SIZE))
    fbo.bind()
    return application, surface, context, fbo


def stats_hold_a_frame(stats):
    """Whether the dictionary says Pika drew a frame: a frame count, the
    draw's GPU time and the memory key usdview shows."""
    return (stats.get("frames", 0) >= 1 and "gpu_draw_ms" in stats
            and stats.get("gpuMemoryUsed", 0) > 0 and stats.get("drawn_triangles", 0) >= 1)


def centre(fbo):
    """The framebuffer's centre pixel as the 8-bit (r, g, b) it holds."""
    pixel = fbo.toImage().pixelColor(SIZE // 2, SIZE // 2)
    return pixel.red(), pixel.green(), pixel.blue()


def scene(path):
    """The engine rendering the scene with Pika, its stage and the params
    of its frames, framed as usdview frames a new stage."""
    stage = Usd.Stage.Open(path)
    engine = UsdImagingGL.Engine()
    plugin = next(p for p in engine.GetRendererPlugins()
                  if engine.GetRendererDisplayName(p) == "Pika")
    if not engine.SetRendererPlugin(plugin):
        sys.exit("engine: cannot use the Pika renderer plugin")
    engine.SetRenderViewport(Gf.Vec4d(0, 0, SIZE, SIZE))
    engine.SetRendererAov("color")
    bounds = UsdGeom.BBoxCache(1, [UsdGeom.Tokens.default_]).ComputeWorldBound(stage.GetPseudoRoot())
    frustum = Gf.Camera().frustum
    frustum.FitToSphere(bounds.ComputeAlignedRange().GetMidpoint(),
                        bounds.ComputeAlignedRange().GetSize().GetLength() / 2)
    engine.SetCameraState(frustum.ComputeViewMatrix(), frustum.ComputeProjectionMatrix())
    params = UsdImagingGL.RenderParams()
    params.frame = Usd.TimeCode(1)
    return engine, stage, params


def frame(engine, stage, params, fbo):
    """Renders once and gives the centre pixel."""
    engine.Render(stage.GetPseudoRoot(), params)
    return centre(fbo)


def pixels(fbo):
    """Every pixel of the framebuffer, as ARGB words."""
    image = fbo.toImage()
    return [image.pixel(x, y) for y in range(SIZE) for x in range(SIZE)]


def capped(path, cap, fbo):
    """The converged image of the scene at `path` in a session of its own
    under the texture cap `cap`, 0 for none, as the host's setting."""
    engine, stage, params = scene(path)
    engine.SetRendererSetting("pika:textureCap", cap)
    converge(engine, stage, params, fbo)
    return pixels(fbo)


def recap(engine, stage, params, fbo, cap):
    """Sets the texture cap to `cap` in the engine's session and renders
    until converged, failing on any frame that builds a material."""
    engine.SetRendererSetting("pika:textureCap", cap)
    deadline = time.monotonic() + PATIENCE
    frames = 0
    while True:
        frame(engine, stage, params, fbo)
        frames += 1
        stats = engine.GetRenderStats()
        building = {k: stats.get(k) for k in ("pending_materials", "compiled_surfaces", "baked_textures")}
        if any(building.values()):
            sys.exit(f"engine: the cap {cap} built a material again: {building}")
        if engine.IsConverged():
            return pixels(fbo), frames
        if time.monotonic() > deadline:
            sys.exit(f"engine: the cap {cap} did not converge in {PATIENCE} s")


def converge(engine, stage, params, fbo):
    """Renders until converged and gives each frame's centre pixel."""
    pixels = []
    deadline = time.monotonic() + PATIENCE
    while True:
        pixels.append(frame(engine, stage, params, fbo))
        if engine.IsConverged():
            return pixels
        if time.monotonic() > deadline:
            sys.exit(f"engine: the scene did not converge in {PATIENCE} s")


def main(clay, painted, mipped):
    keep = gl_context()  # noqa: F841
    fbo = keep[3]
    engine, stage, params = scene(clay)
    before = engine.GetRenderStats()
    if stats_hold_a_frame(before):
        sys.exit(f"engine: the stats held a frame before one was drawn: {before}")
    pixels = converge(engine, stage, params, fbo)
    after = engine.GetRenderStats()
    if not stats_hold_a_frame(after):
        sys.exit(f"engine: the host does not see the frame in Pika's stats: {after}")
    print("engine: the host sees %d stats, %d frames, %.0f triangles drawn, gpuMemoryUsed %.0f" % (
        len(after), after["frames"], after["drawn_triangles"], after["gpuMemoryUsed"]))
    grey = pixels[-1]
    if max(grey) - min(grey) > 1 or grey[0] < 16:
        sys.exit(f"engine: the clay is not grey: {grey}")

    engine, stage, params = scene(painted)
    pixel = frame(engine, stage, params, fbo)
    first = engine.GetRenderStats()
    if engine.IsConverged() or first.get("pending_materials") != 1:
        sys.exit(f"engine: the painted cube's first frame did not wait for its material: {first}")
    if pixel != grey:
        sys.exit(f"engine: the painted cube is {pixel} while its material builds, not clay {grey}")
    pixels = converge(engine, stage, params, fbo)
    last = engine.GetRenderStats()
    if last.get("pending_materials") != 0 or pixels[-1] == grey:
        sys.exit(f"engine: the painted cube stayed clay {pixels[-1]} once built: {last}")
    print("engine: the painted cube was clay %s for %d frames, then %s" % (grey, len(pixels), pixels[-1]))

    # Connecting the surface's diffuse colour to the green node changes the
    # material's structure, so it is built again: meanwhile the cube keeps
    # its look.
    red = pixels[-1]
    surface = UsdShade.Shader.Get(stage, "/world/mtl/paint/surface")
    green = UsdShade.Shader.Get(stage, "/world/mtl/paint/green")
    surface.GetInput("diffuseColor").ConnectToSource(green.ConnectableAPI(), "out")
    # While frames change, a render delivers the frame before and leaves
    # its own on the GPU: the first after the edit delivers the converged
    # frame, and the next delivers the first drawn with the build out.
    frame(engine, stage, params, fbo)
    pixel = frame(engine, stage, params, fbo)
    edited = engine.GetRenderStats()
    if engine.IsConverged() or edited.get("pending_materials") != 1:
        sys.exit(f"engine: the edited cube's first frame did not wait for its material: {edited}")
    if pixel != red:
        sys.exit(f"engine: the edited cube is {pixel} while its material builds, not its {red}: {edited}")
    pixels = converge(engine, stage, params, fbo)
    last = engine.GetRenderStats()
    if last.get("pending_materials") != 0 or pixels[-1] == red or pixels[-1] == grey:
        sys.exit(f"engine: the edited cube stayed {pixels[-1]} once built: {last}")
    print("engine: the edited cube kept %s for %d frames, then %s" % (red, len(pixels), pixels[-1]))

    fresh = {cap: capped(mipped, cap, fbo) for cap in (0, 4, 2)}
    if len({tuple(image) for image in fresh.values()}) != 3:
        sys.exit("engine: the control failed: caps of none, 4 and 2 do not each draw their own image")
    engine, stage, params = scene(mipped)
    converge(engine, stage, params, fbo)
    for cap in (4, 2, 0):
        image, frames = recap(engine, stage, params, fbo, cap)
        if image != fresh[cap]:
            differ = sum(a != b for a, b in zip(image, fresh[cap]))
            sys.exit(f"engine: under the cap {cap}, {differ} pixels differ from a session started at it")
        print(f"engine: the cap changed to {cap} in {frames} frames, building nothing, as a session at it draws")
    recap(engine, stage, params, fbo, 2)
    # The file the raise back to every level reads, cut to its header: the
    # read fails, and the texture is held at its 2×2 levels.
    baked = glob.glob(os.path.join(os.environ["PIKA_CACHE"], "textures", "**", "mips.exr", "*"), recursive=True)
    if not baked:
        sys.exit("engine: mips.exr has no bake under PIKA_CACHE")
    whole = {}
    for path in baked:
        with open(path, "rb") as f:
            whole[path] = f.read()
        os.truncate(path, 40)
    image, frames = recap(engine, stage, params, fbo, 0)
    if image != fresh[2]:
        sys.exit("engine: a raise whose read failed did not keep the 2×2 levels")
    print(f"engine: a raise whose read failed kept the cap 2 image, converged in {frames} frames")
    # Written back in place, as the session reads the file it holds open:
    # the cap changing releases the hold. Unreleased, it would draw the 2×2
    # levels, which differ from the cap 4 image (the control above).
    for path, data in whole.items():
        with open(path, "r+b") as f:
            f.write(data)
    image, frames = recap(engine, stage, params, fbo, 4)
    if image != fresh[4]:
        sys.exit("engine: the cap changing did not release the held raise")
    print(f"engine: the cap changed to 4 released the held raise in {frames} frames")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
