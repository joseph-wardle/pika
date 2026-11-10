"""usdview's --testScript for the gate run (hydra/gate.sh): sizes the
viewport to 1080p at ALab's 3:2, shows the render purpose, repaints until
Pika's stats say a frame was drawn with no material or texture pending
(every build landed, every texture at the levels that fit), then plays frames 1004-1057 twice as the Play button does
(playing on, setFrame, a synchronous repaint). Pika writes the numbers to
PIKA_STATS.
"""

import os
import sys
import time

from PySide6 import QtWidgets

FIRST, COUNT = 1004, 54
WIDTH, HEIGHT = 1620, 1080
PATIENCE = 600


def testUsdviewInputFunction(appController):
    app = QtWidgets.QApplication.instance()
    view = appController._stageView
    # ALab's textures are 145 GB at their own size; GATE_CAP=512 is the run to compare with.
    cap = int(os.environ.get("GATE_CAP", "0"))
    if cap:
        view.SetRendererSetting("pika:textureCap", cap)
    view.SetPhysicalWindowSize(WIDTH, HEIGHT)
    settings = appController._dataModel.viewSettings
    settings.displayRender = True
    settings.displayProxy = False
    view.repaint()
    app.processEvents()
    # The delegate's IsConverged is true before Pika has drawn at all.
    start = time.monotonic()
    while True:
        stats = view._renderer.GetRenderStats()
        pending = stats.get("pending_materials", 0), stats.get("pending_textures", 0)
        if stats.get("frames", 0) and not any(pending):
            break
        if time.monotonic() - start > PATIENCE:
            sys.exit(f"{pending[0]} materials and {pending[1]} textures still pending after {PATIENCE} s")
        view.repaint()
        app.processEvents()
    print(
        f"every material built after {time.monotonic() - start:.1f} s and {stats['frames']:.0f} frames",
        flush=True,
    )
    appController._dataModel.playing = True
    for _ in range(2):
        for frame in range(FIRST, FIRST + COUNT):
            appController.setFrame(frame)
            view.repaint()
            app.processEvents()
    appController._dataModel.playing = False
