"""Writes the bake check's source images once. They are tracked.

    python3 bake/fixtures/textures.py

ramp.exr    8×8, two float channels, scanline. Texel (x, y) holds its own centre,
            ((x + 0.5) / 8, (y + 0.5) / 8), so a sample returns where it was taken.
swatch.png  8×8 RGB at 8 bits, four flat 4×4 blocks of red, green, blue and mid grey. Flat
            blocks encode exactly, so a BC7 bake gives the bytes back.
mips.exr    8×8 RGB half, tiled with stored levels as maketx writes them. Level 0 is the
            swatch, the rest flat greys of 128, 64 and 32 of 255, which no filter of level 0
            gives, so reused levels are told from rebuilt ones and a render shows the level read.
normal.exr  4×4, three float channels, each texel (0.2, 0.6, 0.898), encoded as
            (153, 204, 242) / 255 so an 8-bit bake is exact at any level.
normal.png  The same at 8 bits, as a painting tool exports it. A vector3 image declaring no
            colour space is read as it is, not as sRGB.
greys.jpg   128×8 JPEG at quality 100, sixteen flat 8×8 greys in a row, 0, 17, 34 … 255. A
            flat block is one DC coefficient with centred chroma, so it decodes exactly, and the
            generator refuses one that doesn't.
acescg.exr  32×4, three float channels, eight flat 4×4 blocks of COLORS in ACEScg primaries,
            so a bake declared acescg gives COLORS back.
udim.<UDIM>.exr
            Tiles 1001, the ramp, and 1012, the ramp at half scale, with a gap each way, so a
            sample tells the tiles apart and the ones around them are missing.
wrapped.exr 8×8, three float channels, four flat 4×4 blocks of WRAPPED, multiples of 1/255 so a
            linear 8-bit bake is exact. Its wrapmodes attribute states "periodic,clamp", as maketx
            writes it, for a UsdUVTexture's useMetadata.
mixed.<UDIM>.exr
            Tiles of different sizes, as ALab's stoat has: 1001 the ramp, 1002 a 4×4 ramp, which
            a bake enlarges bilinearly to 8×8, the outer half texel clamped.
sky.exr     64×32 float latlong, the texel toward d holding 0.5 + 0.4 d, so a turned or flipped
            sky shows.
spot.exr    64×32 float latlong of grey 0.3 with one texel of SPOT, 59° up and 20° round from +z
            toward +x, a sun for a dome light to lift.
ripples.exr 1024×512 half latlong of grey 0.5 rippled by ±RIPPLE every RIPPLE_PERIOD texels round,
            with a disc of DISC toward TOWARD, a sun for a dome light to lift. Each halving
            flattens the ripple further, so a render shows the levels it blends, and a quarter
            turn moves it.
checker.exr 512×256 half latlong of 2×2 squares of 0.2 and 0.8, a sky whose edges show where a
            frame's history lags.
"""

import os

import numpy as np
import OpenImageIO as oiio

HERE = os.path.dirname(os.path.abspath(__file__))
SWATCH = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)]
# Neighbouring blocks differ by at most 0.2 in linear light: the GPU's 8-bit blend
# weights miss hydra/image.py's tolerance at larger steps.
GREYS = [17 * i for i in range(16)]
WRAPPED = [51, 102, 153, 204]
SPOT = 400.0
DISC = 40.0
TOWARD = (14, 8, -12)
# Its angular radius, in degrees.
DISC_RADIUS = 4
# Gentle enough that rays a third of a pixel apart differ by less than hydra/image.py's STEP.
RIPPLE = 0.15
RIPPLE_PERIOD = 12
COLORS = [(240, 60, 60), (220, 100, 60), (200, 140, 60), (180, 180, 60),
          (140, 200, 80), (100, 200, 120), (60, 200, 160), (60, 180, 200)]
# Linear Rec.709 to ACEScg: the inverse of hydra/image.py's ACESCG_TO_REC709.
REC709_TO_ACESCG = np.linalg.inv([[1.705051, -0.621792, -0.083259],
                                  [-0.130256, 1.140805, -0.010548],
                                  [-0.024003, -0.128969, 1.152972]])


def blocks(cell, colors, columns, dtype):
    rows = len(colors) // columns
    pixels = np.zeros((rows * cell, columns * cell, 3), dtype)
    for i, color in enumerate(colors):
        y, x = cell * (i // columns), cell * (i % columns)
        pixels[y:y + cell, x:x + cell] = color
    return pixels


def linear(srgb):
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


def wrapped():
    pixels = blocks(4, [(g, g, g) for g in WRAPPED], 2, np.float32) / 255
    path = os.path.join(HERE, "wrapped.exr")
    spec = oiio.ImageSpec(8, 8, 3, oiio.FLOAT)
    spec.attribute("wrapmodes", "periodic,clamp")
    out = oiio.ImageOutput.create(path)
    out.open(path, spec)
    out.write_image(np.ascontiguousarray(pixels))
    out.close()
    if oiio.ImageInput.open(path).spec().getattribute("wrapmodes") != "periodic,clamp":
        raise SystemExit("wrapped.exr does not keep its wrapmodes")


def normal():
    codes = np.broadcast_to(np.array([153, 204, 242], np.uint8), (4, 4, 3))
    out = oiio.ImageOutput.create("normal.exr")
    out.open(os.path.join(HERE, "normal.exr"), oiio.ImageSpec(4, 4, 3, oiio.FLOAT))
    out.write_image(np.ascontiguousarray(codes.astype(np.float32) / 255))
    out.close()
    out = oiio.ImageOutput.create("normal.png")
    out.open(os.path.join(HERE, "normal.png"), oiio.ImageSpec(4, 4, 3, oiio.UINT8))
    out.write_image(np.ascontiguousarray(codes))
    out.close()


def write_sky(name, pixels):
    out = oiio.ImageOutput.create(name)
    out.open(os.path.join(HERE, name), oiio.ImageSpec(64, 32, 3, oiio.FLOAT))
    out.write_image(np.ascontiguousarray(pixels.astype(np.float32)))
    out.close()


def directions(width, height):
    # shaders/view.slang's from_latlong.
    y, x = np.mgrid[:height, :width]
    longitude, latitude = (0.5 - (x + 0.5) / width) * 2 * np.pi, (0.5 - (y + 0.5) / height) * np.pi
    c = np.cos(latitude)
    return np.stack([c * np.sin(longitude), np.sin(latitude), c * np.cos(longitude)], -1)


def sky():
    write_sky("sky.exr", 0.5 + 0.4 * directions(64, 32))


def checker():
    y, x = np.mgrid[:256, :512]
    pixels = np.where(((x // 2 + y // 2) % 2)[..., None] == 1, 0.8, 0.2) * np.ones(3)
    out = oiio.ImageOutput.create("checker.exr")
    spec = oiio.ImageSpec(512, 256, 3, oiio.HALF)
    spec.attribute("compression", "zip")
    out.open(os.path.join(HERE, "checker.exr"), spec)
    out.write_image(np.ascontiguousarray(pixels, np.float32))
    out.close()


def ripples():
    toward = np.array(TOWARD) / np.linalg.norm(TOWARD)
    within = directions(1024, 512) @ toward > np.cos(np.radians(DISC_RADIUS))
    x = np.arange(1024) + 0.5
    rippled = np.broadcast_to(0.5 + RIPPLE * np.sin(2 * np.pi * x / RIPPLE_PERIOD), (512, 1024))
    pixels = np.where(within, DISC, rippled)[..., None] * np.ones(3)
    path = os.path.join(HERE, "ripples.exr")
    spec = oiio.ImageSpec(1024, 512, 3, oiio.HALF)
    spec.attribute("compression", "zip")
    out = oiio.ImageOutput.create(path)
    out.open(path, spec)
    out.write_image(np.ascontiguousarray(pixels, np.float32))
    out.close()
    read = oiio.ImageBuf(path)
    if read.has_error or np.abs(read.get_pixels(oiio.FLOAT) - pixels).max() > 1e-2:
        raise SystemExit("ripples.exr does not read back as written")


def spot():
    pixels = np.full((32, 64, 3), 0.3)
    pixels[5, 28] = SPOT
    write_sky("spot.exr", pixels)


if __name__ == "__main__":
    ramp()
    swatch()
    mips()
    wrapped()
    normal()
    greys()
    acescg()
    udim()
    mixed()
    sky()
    spot()
    ripples()
    checker()
