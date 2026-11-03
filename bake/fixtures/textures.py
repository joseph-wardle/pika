"""Writes the bake check's source images, once; they are tracked.

    hython bake/fixtures/textures.py

ramp.exr    8×8, two float channels, a scanline EXR: texel (x, y) holds its
            own centre, ((x + 0.5) / 8, (y + 0.5) / 8), so a sample inside
            the image returns the coordinate it was taken at.
swatch.png  8×8 RGB at 8 bits, four flat 4×4 blocks: red, green, blue and
            mid grey. Flat blocks encode exactly, so a BC7 bake must give the
            bytes back.
mips.exr    8×8 RGB half, tiled with stored mip levels, as maketx writes
            them: level 0 is the swatch, the coarser levels are flat greys of
            0.5, 0.25 and 0.125, which no filter of level 0 produces, so a
            bake that reuses the stored levels is told from one that rebuilt
            them.
normal.exr  4×4, three float channels, every texel the same tangent-space
            normal, (0.2, 0.6, 0.898) encoded as (153, 204, 242) / 255, so
            Pika's 8-bit bake of it is exact and a sample anywhere, at any
            level, returns it (hydra/images/normalmap.usda).
"""

import os

import numpy as np
import OpenImageIO as oiio

HERE = os.path.dirname(os.path.abspath(__file__))
SWATCH = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)]




def ramp():
    coords = np.zeros((8, 8, 2), np.float32)
    for y in range(8):
        for x in range(8):
            coords[y, x] = ((x + 0.5) / 8, (y + 0.5) / 8)
    spec = oiio.ImageSpec(8, 8, 2, oiio.FLOAT)
    spec.channelnames = ("R", "G")
    out = oiio.ImageOutput.create("ramp.exr")
    out.open(os.path.join(HERE, "ramp.exr"), spec)
    out.write_image(coords)
    out.close()


def swatch_pixels():
    pixels = np.zeros((8, 8, 3), np.uint8)
    for i, color in enumerate(SWATCH):
        pixels[4 * (i // 2):4 * (i // 2) + 4, 4 * (i % 2):4 * (i % 2) + 4] = color
    return pixels


def swatch():
    out = oiio.ImageOutput.create("swatch.png")
    out.open(os.path.join(HERE, "swatch.png"), oiio.ImageSpec(8, 8, 3, oiio.UINT8))
    out.write_image(swatch_pixels())
    out.close()


def mips():
    levels = [swatch_pixels().astype(np.float32) / 255]
    for size, grey in ((4, 0.5), (2, 0.25), (1, 0.125)):
        levels.append(np.full((size, size, 3), grey, np.float32))
    out = oiio.ImageOutput.create("mips.exr")
    path = os.path.join(HERE, "mips.exr")
    for i, level in enumerate(levels):
        spec = oiio.ImageSpec(level.shape[1], level.shape[0], 3, oiio.HALF)
        spec.tile_width = spec.tile_height = 8
        spec.attribute("textureformat", "Plain Texture")
        mode = "Create" if i == 0 else "AppendMIPLevel"
        if not out.open(path, spec, mode):
            raise SystemExit(out.geterror())
        out.write_image(level)
    out.close()


def normal():
    texels = np.broadcast_to(np.array([153, 204, 242], np.float32) / 255, (4, 4, 3))
    out = oiio.ImageOutput.create("normal.exr")
    out.open(os.path.join(HERE, "normal.exr"), oiio.ImageSpec(4, 4, 3, oiio.FLOAT))
    out.write_image(np.ascontiguousarray(texels))
    out.close()


ramp()
swatch()
mips()
normal()
