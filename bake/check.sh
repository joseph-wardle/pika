#!/usr/bin/env bash
# The bake tool's check, run where Houdini 22 and slangc are (not in CI).
#
# Builds the tool and generates bake/fixtures/network.mtlx, the document
# src/material.rs writes for hydra/fixtures/network.usda. The manifest must
# match bake/fixtures/network.manifest, and the pixel stage must compile
# with network.surface.slang, the vertex stage src/surface.rs writes for
# it, as Pika compiles them, reflecting the layout in
# network.reflection.json that src/surface.rs's tests pack against. A
# document naming a nodedef the libraries lack must fail, naming it.
#
#   bake/check.sh
set -euo pipefail

HFS=${HFS:-/opt/hfs22.0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/target/bake/check
BAKE=$ROOT/target/bake/bake

make -C "$ROOT/bake" --no-print-directory HFS="$HFS"
rm -rf "$OUT"
mkdir -p "$OUT/network" "$OUT/missing" "$OUT/textures"
FIXTURES=$ROOT/bake/fixtures

failed=0
fail() {
    echo "FAIL: $*"
    failed=1
}

"$BAKE" generate "$ROOT/bake/fixtures/network.mtlx" "$OUT/network" ||
    fail "network: generate failed"
diff "$ROOT/bake/fixtures/network.manifest" "$OUT/network/manifest" ||
    fail "network: the manifest differs from the reference"
# The generator declares textures inside its uniform block, and slangc warns
# that it moves them to bindings of their own, which is where Pika binds them.
cp "$ROOT/bake/fixtures/network.surface.slang" "$OUT/network/surface.slang"
slangc "$OUT/network/surface.slang" -I "$ROOT/shaders" -target spirv -fvk-use-entrypoint-name \
    -warnings-as-errors all -warnings-disable 31106,31107 -o "$OUT/network/surface.spv" \
    -reflection-json "$OUT/network/reflection.json" ||
    fail "network: slangc rejected the surface"
diff "$ROOT/bake/fixtures/network.reflection.json" "$OUT/network/reflection.json" ||
    fail "network: the reflection differs from the reference"

sed 's/ND_geompropvalue_vector2/ND_missing/' "$ROOT/bake/fixtures/network.mtlx" > "$OUT/missing.mtlx"
if "$BAKE" generate "$OUT/missing.mtlx" "$OUT/missing" 2> "$OUT/missing.err"; then
    fail "missing: a document with an unknown nodedef generated"
fi
grep -q "ND_missing" "$OUT/missing.err" || fail "missing: the error does not name the nodedef"

texture() {
    "$BAKE" texture "$FIXTURES/$1" "$2" "$3" "$4" "" "$OUT/textures/$5.texture" || fail "$5: texture failed"
}
texture ramp.exr raw float 4096 ramp
cmp "$FIXTURES/ramp.texture" "$OUT/textures/ramp.texture" || fail "ramp: the bake differs from the reference"
texture swatch.png srgb_texture color 4096 swatch
texture swatch.png raw color 4096 swatch_raw
texture swatch.png raw single 4096 single
texture mips.exr raw data 4 mips
texture mips.exr raw data 8 mips8
# "auto" is sRGB for an 8-bit image and raw for a float one.
texture swatch.png auto color 4096 swatch_auto
cmp "$OUT/textures/swatch.texture" "$OUT/textures/swatch_auto.texture" || fail "swatch_auto: auto is not sRGB for a PNG"
texture ramp.exr auto float 4096 ramp_auto
cmp "$FIXTURES/ramp.texture" "$OUT/textures/ramp_auto.texture" || fail "ramp_auto: auto is not raw for an EXR"
# A JPEG reads, and a space the config knows beyond sRGB is converted.
texture greys.jpg auto color 4096 greys
texture acescg.exr acescg color 4096 acescg
! cmp -s "$OUT/textures/swatch.texture" "$OUT/textures/swatch_raw.texture" || fail "swatch_raw: raw and sRGB bake alike, so the auto checks prove nothing"
"$HFS/bin/hython" "$ROOT/bake/texture.py" "$OUT/textures" || fail "textures: a bake does not decode to its source"
if "$BAKE" texture "$FIXTURES/swatch.png" bogus color 4096 "" "$OUT/textures/bogus.texture" 2> "$OUT/textures/bogus.err"; then
    fail "bogus: a texture in an unknown colour space baked"
fi
grep -q "bogus" "$OUT/textures/bogus.err" || fail "bogus: the error does not name the colour space"

[ $failed = 0 ] && echo "PASS: bake"
exit $failed
