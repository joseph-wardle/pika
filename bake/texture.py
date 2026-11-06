"""The bake check's texture oracle: decodes the files `bake texture` wrote
from the fixtures in bake/fixtures/ (textures.py says what they hold) and
compares them with what the sources say they must decode to.

    hython bake/texture.py <out dir>

Block-compressed levels are decoded by OpenImageIO through a DDS wrapper
written beside each file, so the check never trusts Pika's own decoding.
"""

import os
import struct
import sys

import numpy as np
import OpenImageIO as oiio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures"))
import textures  # noqa: E402

# VkFormat to DXGI_FORMAT, and the channel count of the float formats.
DXGI = {139: 80, 145: 98, 146: 99}
FLOAT = {100: 1, 103: 2, 109: 4}
failed = False


def fail(what):
    global failed
    failed = True
    print("FAIL:", what)


def header(path):
    """The header's five numbers, the source's size, the wrap modes (u, v),
    the tile table and the levels' bytes."""
    with open(path, "rb") as f:
        bytes = f.read()
    magic, width, height, layers, levels, fmt, *rest = struct.unpack("<4s10I", bytes[:44])
    if magic != b"pika":
        raise ValueError(f"{path} is not a texture file")
    source, wrap, entries = tuple(rest[:2]), tuple(rest[2:4]), rest[4]
    table = struct.unpack(f"<{entries}I", bytes[44:44 + 4 * entries])
    return width, height, layers, levels, fmt, source, wrap, table, bytes[44 + 4 * entries:]


def decode(path):
    """Each level as an array of shape (height, width, channels), or for a
    float texture of several layers (layers, height, width, channels)."""
    width, height, layers, levels, fmt, _, _, _, data = header(path)
    if fmt in FLOAT:
        out, offset, w, h = [], 0, width, height
        for _ in range(levels):
            n = layers * w * h * FLOAT[fmt]
            level = np.frombuffer(data[offset:offset + 4 * n], np.float32).reshape(layers, h, w, FLOAT[fmt])
            out.append(level if layers > 1 else level[0])
            offset, w, h = offset + 4 * n, max(1, w // 2), max(1, h // 2)
        return out
    if layers != 1:
        raise ValueError(f"{path} has {layers} layers")
    dds = path + ".dds"
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000
    caps = 0x1000 | 0x400000 | 0x8
    with open(dds, "wb") as f:
        f.write(b"DDS ")
        f.write(struct.pack("<7I11I", 124, flags, height, width, len(data), 1, levels, *([0] * 11)))
        f.write(struct.pack("<2I4s5I", 32, 0x4, b"DX10", 0, 0, 0, 0, 0))
        f.write(struct.pack("<5I", caps, 0, 0, 0, 0))
        f.write(struct.pack("<5I", DXGI[fmt], 3, 0, 1, 0))
        f.write(data)
    image = oiio.ImageInput.open(dds)
    if image is None:
        raise ValueError(f"{dds}: {oiio.geterror()}")
    out = []
    while image.seek_subimage(0, len(out)):
        out.append(image.read_image(0, len(out), 0, image.spec().nchannels, oiio.UINT8))
    image.close()
    return out


def expect(name, level, want, tolerance):
    got = np.asarray(level, np.float64)
    want = np.asarray(want, np.float64)
    if got.shape != want.shape:
        return fail(f"{name}: shape {got.shape}, not {want.shape}")
    if np.abs(got - want).max() > tolerance:
        fail(f"{name}: differs by up to {np.abs(got - want).max()} (allowed {tolerance})")


def srgb(linear):
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)


def linear(srgb):
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def check(out):
    fixtures = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    swatch = np.asarray(oiio.ImageBuf(os.path.join(fixtures, "swatch.png")).get_pixels(oiio.UINT8))

    def baked(name, want, source=None, wrap=(0, 0), table=(0,)):
        path = os.path.join(out, name + ".texture")
        got = header(path)[:8]
        want = want + (source or want[:2], wrap, table)
        if got != want:
            fail(f"{name}: the header says {got}, not {want}")
        return decode(path)

    # Floats pass through; the levels below the source are box filtered.
    levels = baked("ramp", (8, 8, 1, 4, 103))
    for i, level in enumerate(levels):
        size = 8 >> i
        want = [[((x + 0.5) / size, (y + 0.5) / size) for x in range(size)] for y in range(size)]
        expect(f"ramp level {i}", level, want, 0)

    # Flat blocks come back within a step of BC7, sRGB in and sRGB out; the
    # filtered levels average in linear light.
    levels = baked("swatch", (8, 8, 1, 4, 146))
    expect("swatch level 0", levels[0][..., :3], swatch, 2)
    expect("swatch level 0 alpha", levels[0][..., 3], np.full((8, 8), 255), 2)
    expect("swatch level 1", levels[1][..., :3], swatch[::2, ::2], 2)
    mean = linear(swatch / 255).reshape(-1, 3).mean(0)
    expect("swatch level 3", levels[3][0, 0, :3], srgb(mean) * 255, 2)

    # The control: the same bytes declared linear are another texture.
    levels = baked("swatch_raw", (8, 8, 1, 4, 146))
    if np.abs(levels[0][..., :3].astype(int) - swatch).max() < 16:
        fail("swatch_raw: the declared colour space made no difference")

    # An 8-bit JPEG declared auto is sRGB, so its greys come back as
    # stored; the colours stored in ACEScg primaries come back as sRGB codes.
    levels = baked("greys", (128, 8, 1, 8, 146))
    expect("greys level 0", levels[0][..., :3], textures.blocks(8, [(g, g, g) for g in textures.GREYS], 16, np.uint8), 0)
    levels = baked("acescg", (32, 4, 1, 6, 146))
    expect("acescg level 0", levels[0][..., :3], textures.blocks(4, textures.COLORS, 8, np.uint8), 1)

    # A set's tiles are its layers in tile order, and the table sends tile
    # 1001 + i to its layer, the tiles the set lacks past the end.
    table = [2 ** 32 - 1] * 100
    table[0], table[11] = 0, 1
    levels = baked("udim", (8, 8, 2, 4, 103), table=tuple(table))
    expect("udim level 0 tile 1001", levels[0][0], textures.ramp_pixels(1.0), 0)
    expect("udim level 0 tile 1012", levels[0][1], textures.ramp_pixels(0.5), 0)
    expect("udim level 3 tile 1012", levels[3][1], [[(0.25, 0.25)]], 0)

    # A smaller tile is enlarged to the set's size bilinearly, texel centres
    # aligned: the ramp's texels then hold their own centres again, but for
    # the outer half texel, which clamps to the edge.
    table[1], table[11] = 1, 2 ** 32 - 1
    levels = baked("mixed", (8, 8, 2, 4, 103), table=tuple(table))
    expect("mixed level 0 tile 1001", levels[0][0], textures.ramp_pixels(1.0), 0)
    centre = (np.clip((np.arange(8) + 0.5) / 2 - 0.5, 0, 3) + 0.5) / 4
    want = np.stack(np.meshgrid(centre, centre), -1)
    expect("mixed level 0 tile 1002", levels[0][1], want, 1e-6)

    # The wrap modes a file states are kept, u then v; one stating none
    # (the ramp, the swatch) bakes black.
    levels = baked("wrapped", (8, 8, 1, 4, 145), wrap=(2, 1))
    expect("wrapped level 0", levels[0][..., :3], textures.blocks(4, [(g, g, g) for g in textures.WRAPPED], 2, np.uint8), 0)

    levels = baked("single", (8, 8, 1, 4, 139))
    expect("single level 0", levels[0][..., 0], swatch[..., 0], 1)

    # The stored levels are used from the first within the cap, the header
    # keeping the source's size; those a filter would produce differ from
    # them by construction (textures.py).
    levels = baked("mips", (4, 4, 1, 3, 145), source=(8, 8))
    for i, (level, grey) in enumerate(zip(levels, (128, 64, 32))):
        expect(f"mips level {i}", level[..., :3], np.full(level.shape[:2] + (3,), grey), 1)
    levels = baked("mips8", (8, 8, 1, 4, 145))
    expect("mips8 level 0", levels[0][..., :3], swatch, 2)
    expect("mips8 level 1", levels[1][..., :3], np.full((4, 4, 3), 128), 1)


if __name__ == "__main__":
    check(sys.argv[1])
    sys.exit(1 if failed else 0)
