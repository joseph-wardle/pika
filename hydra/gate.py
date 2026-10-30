"""usdview's --testScript for the gate run (hydra/gate.sh): sizes the
viewport to 1080p at ALab's 3:2, shows the render purpose, then plays
frames 1004-1057 twice as the Play button does (playing on, setFrame, a
synchronous repaint). Pika writes the numbers to PIKA_STATS.
"""

from PySide6 import QtWidgets

FIRST, COUNT = 1004, 54
WIDTH, HEIGHT = 1620, 1080


def testUsdviewInputFunction(appController):
    app = QtWidgets.QApplication.instance()
    view = appController._stageView
    view.SetPhysicalWindowSize(WIDTH, HEIGHT)
    settings = appController._dataModel.viewSettings
    settings.displayRender = True
    settings.displayProxy = False
    view.repaint()
    app.processEvents()
    appController._dataModel.playing = True
    for _ in range(2):
        for frame in range(FIRST, FIRST + COUNT):
            appController.setFrame(frame)
            view.repaint()
            app.processEvents()
    appController._dataModel.playing = False
