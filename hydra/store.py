"""Renders a fixture's frame 1, runs its edit and renders frame 2 through UsdImagingGL, standing in
for husk where there is none, so that hydra/check.sh can compare the stores Pika dumps.

    python3 hydra/store.py <scene.usda> [edit.py]

Needs a display and the delegate on PXR_PLUGINPATH_NAME.
"""

import os
import sys

from pxr import Gf, Usd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import converge, gl_context, scene  # noqa: E402


def main(path, edit=None):
    keep = gl_context()  # noqa: F841
    fbo = keep[3]
    engine, stage, params = scene(path)
    # Any camera does, as the store holds what Hydra hands over, seen or not.
    frustum = Gf.Frustum()
    engine.SetCameraState(frustum.ComputeViewMatrix(), frustum.ComputeProjectionMatrix())
    params.showRender = True
    params.showProxy = False
    converge(engine, stage, params, fbo)
    if edit:
        with open(edit) as f:
            exec(f.read(), {"stage": stage})
    params.frame = Usd.TimeCode(2)
    converge(engine, stage, params, fbo)


if __name__ == "__main__":
    main(*sys.argv[1:])
