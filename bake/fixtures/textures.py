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
            128, 64 and 32 of 255, which no filter of level 0 produces, so a
            bake that reuses the stored levels is told from one that rebuilt
            them, and which are exact at 8 bits, so a render shows which
            levels the GPU read (hydra/images/mips.usda).
normal.exr  4×4, three float channels, every texel the same tangent-space
            normal, (0.2, 0.6, 0.898) encoded as (153, 204, 242) / 255, so
            Pika's 8-bit bake of it is exact and a sample anywhere, at any
            level, returns it (hydra/images/normalmap.usda).
greys.jpg   128×8 RGB JPEG at quality 100, sixteen flat 8×8 blocks of grey
            in a row, 0, 17, 34 … 255. A flat grey block is one DC
            coefficient with centred chroma, so the JPEG decodes exactly,
            and the generator refuses to write one that does not.
acescg.exr  32×4, three float channels, eight flat 4×4 blocks of COLORS in
            a row, linear and in ACEScg primaries, the inverse of the matrix
            hydra/image.py converts an acescg image by, so a bake declared
            acescg gives COLORS back.
udim.<UDIM>.exr
            A two-tile UDIM set with a gap in each direction: tile 1001 is
            the ramp, tile 1012 the ramp at half scale, texel (x, y) holding
            ((x + 0.5) / 16, (y + 0.5) / 16), so a sample tells the tiles
            apart and the tiles between and around them are missing
            (hydra/images/udim.usda).
mixed.<UDIM>.exr
            A set whose tiles differ in size, as ALab's stoat has: tile 1001
            is the ramp, tile 1002 a 4×4 ramp, texel (x, y) holding
            ((x + 0.5) / 4, (y + 0.5) / 4), which a bake enlarges to the
            set's 8×8 bilinearly, so its texels hold the ramp at their own
            centres, the outer half texel clamped.
"""

import os

import numpy as np
import OpenImageIO as oiio

HERE = os.path.dirname(os.path.abspath(__file__))
SWATCH = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)]
# Neighbouring blocks of these differ by at most 0.2 in linear light: the
# GPU blends texels with 8-bit weights, so a larger step misses
# hydra/image.py's tolerance where two blocks blend.
GREYS = [17 * i for i in range(16)]
COLORS = [(240, 60, 60), (220, 100, 60), (200, 140, 60), (180, 180, 60),
          (140, 200, 80), (100, 200, 120), (60, 200, 160), (60, 180, 200)]
# Linear Rec.709 to ACEScg: the inverse of hydra/image.py's ACESCG_TO_REC709.
REC709_TO_ACESCG = np.linalg.inv([[1.705051, -0.621792, -0.083259],
                                  [-0.130256, 1.140805, -0.010548],
                                  [-0.024003, -0.128969, 1.152972]])


def blocks(cell, colors, columns, dtype):
    """Flat cell×cell blocks, one per colour, in reading order over `columns`."""
    rows = len(colors) // columns
    pixels = np.zeros((rows * cell, columns * cell, 3), dtype)
    for i, color in enumerate(colors):
        y, x = cell * (i // columns), cell * (i % columns)
        pixels[y:y + cell, x:x + cell] = color
    return pixels


def linear(srgb):
    """Linear light from sRGB codes."""
    v = np.asarray(srgb) / 255
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def ramp_pixels(scale, size=8):
    coords = np.zeros((size, size, 2), np.float32)
    for y in range(size):
        for x in range(size):
            coords[y, x] = ((x + 0.5) / size * scale, (y + 0.5) / size * scale)
    return coords


def write_ramp(name, scale, size=8):
    spec = oiio.ImageSpec(size, size, 2, oiio.FLOAT)
    spec.channelnames = ("R", "G")
    out = oiio.ImageOutput.create(name)
    out.open(os.path.join(HERE, name), spec)
    out.write_image(ramp_pixels(scale, size))
    out.close()


def ramp():
    write_ramp("ramp.exr", 1.0)


def udim():
    write_ramp("udim.1001.exr", 1.0)
    write_ramp("udim.1012.exr", 0.5)


def mixed():
    write_ramp("mixed.1001.exr", 1.0)
    write_ramp("mixed.1002.exr", 1.0, 4)


def swatch_pixels():
    return blocks(4, SWATCH, 2, np.uint8)


def swatch():
    out = oiio.ImageOutput.create("swatch.png")
    out.open(os.path.join(HERE, "swatch.png"), oiio.ImageSpec(8, 8, 3, oiio.UINT8))
    out.write_image(swatch_pixels())
    out.close()


def greys():
    pixels = blocks(8, [(g, g, g) for g in GREYS], 16, np.uint8)
    spec = oiio.ImageSpec(128, 8, 3, oiio.UINT8)
    spec.attribute("CompressionQuality", 100)
    path = os.path.join(HERE, "greys.jpg")
    out = oiio.ImageOutput.create(path)
    out.open(path, spec)
    out.write_image(pixels)
    out.close()
    if (np.asarray(oiio.ImageBuf(path).get_pixels(oiio.UINT8)) != pixels).any():
        raise SystemExit("greys.jpg does not decode to its greys")


def acescg():
    pixels = blocks(4, [REC709_TO_ACESCG @ linear(c) for c in COLORS], 8, np.float32)
    out = oiio.ImageOutput.create("acescg.exr")
    out.open(os.path.join(HERE, "acescg.exr"), oiio.ImageSpec(32, 4, 3, oiio.FLOAT))
    out.write_image(pixels)
    out.close()


def mips():
    levels = [swatch_pixels().astype(np.float32) / 255]
    for size, grey in ((4, 128), (2, 64), (1, 32)):
        levels.append(np.full((size, size, 3), grey / 255, np.float32))
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


if __name__ == "__main__":
    ramp()
    swatch()
    mips()
    normal()
    greys()
    acescg()
    udim()
    mixed()
