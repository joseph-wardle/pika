"""Renders a scene through UsdImagingGL in this process, as usdview does,
and checks what the host then sees of Pika: the render stats dictionary
with the last frame's pairs and gpuMemoryUsed, which usdview's HUD shows.

    hython hydra/engine.py <scene.usda>

Needs a display (check.sh runs it on its Xvnc) and the delegate on
PXR_PLUGINPATH_NAME. The control is the dictionary before the first frame:
it has no frame pairs, so the check must fail on it.
"""

import sys

from PySide6.QtCore import QSize
from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLFramebufferObject
from PySide6.QtWidgets import QApplication
from pxr import Gf, Usd, UsdGeom, UsdImagingGL

SIZE = 64


def gl_context():
    """A current GL context with a bound framebuffer, which the engine's
    present task needs even though the frame never reaches a window."""
    application = QApplication(sys.argv)
    surface = QOffscreenSurface()
    surface.setFormat(QSurfaceFormat())
    surface.create()
    context = QOpenGLContext()
    context.create()
    context.makeCurrent(surface)
    fbo = QOpenGLFramebufferObject(QSize(1, 1))
    fbo.bind()
    return application, surface, context, fbo


def stats_hold_a_frame(stats):
    """Whether the dictionary says Pika drew a frame: a frame count, the
    draw's GPU time and the memory key usdview shows."""
    return (stats.get("frames", 0) >= 1 and "gpu_draw_ms" in stats
            and stats.get("gpuMemoryUsed", 0) > 0 and stats.get("drawn_triangles", 0) >= 1)


def main(path):
    keep = gl_context()  # noqa: F841
    stage = Usd.Stage.Open(path)
    engine = UsdImagingGL.Engine()
    plugin = next(p for p in engine.GetRendererPlugins()
                  if engine.GetRendererDisplayName(p) == "Pika")
    if not engine.SetRendererPlugin(plugin):
        sys.exit("engine: cannot use the Pika renderer plugin")
    engine.SetRenderViewport(Gf.Vec4d(0, 0, SIZE, SIZE))
    engine.SetRendererAov("color")
    # A camera framing the stage's bounds, as usdview frames a new stage.
    bounds = UsdGeom.BBoxCache(1, [UsdGeom.Tokens.default_]).ComputeWorldBound(stage.GetPseudoRoot())
    frustum = Gf.Camera().frustum
    frustum.FitToSphere(bounds.ComputeAlignedRange().GetMidpoint(),
                        bounds.ComputeAlignedRange().GetSize().GetLength() / 2)
    engine.SetCameraState(frustum.ComputeViewMatrix(), frustum.ComputeProjectionMatrix())
    params = UsdImagingGL.RenderParams()
    params.frame = Usd.TimeCode(1)

    before = engine.GetRenderStats()
    if stats_hold_a_frame(before):
        sys.exit(f"engine: the stats held a frame before one was drawn: {before}")
    for _ in range(10):
        engine.Render(stage.GetPseudoRoot(), params)
        if engine.IsConverged():
            break
    after = engine.GetRenderStats()
    if not stats_hold_a_frame(after):
        sys.exit(f"engine: the host does not see the frame in Pika's stats: {after}")
    print("engine: the host sees %d stats, %d frames, %.0f triangles drawn, gpuMemoryUsed %.0f" % (
        len(after), after["frames"], after["drawn_triangles"], after["gpuMemoryUsed"]))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
