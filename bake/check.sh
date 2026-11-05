#!/usr/bin/env bash
# The bake tool's check, run where Houdini 22 and slangc are (not in CI).
#
# Builds the tool and generates bake/fixtures/network.mtlx, the document
# src/material.rs writes for hydra/fixtures/network.usda. The manifest must
# match bake/fixtures/network.manifest, and the surface must compile as
# Pika compiles it, reflecting the layout in network.reflection.json that
# src/surface.rs's tests pack against. A
# document naming a nodedef the libraries lack, or reading a primvar the
# renderer's vertices lack, or UV0 beside another primvar, must fail,
# naming it. A surface reading UV0 alone reads the texcoord. The
# libraries job must list the manifest's MaterialX version and the standard
# library's definitions with their size and time.
#
#   bake/check.sh
set -euo pipefail

HFS=${HFS:-/opt/hfs22.0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/target/bake/check
BAKE=$ROOT/target/bake/bake

make -C "$ROOT/bake" --no-print-directory HFS="$HFS"
rm -rf "$OUT"
mkdir -p "$OUT/network" "$OUT/missing" "$OUT/primvar" "$OUT/uv0" "$OUT/perfuv" "$OUT/textures"
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
slangc "$OUT/network/surface.slang" -I "$ROOT/shaders" -target spirv -fvk-use-entrypoint-name \
    -warnings-as-errors all -warnings-disable 31106,31107 -o "$OUT/network/surface.spv" \
    -reflection-json "$OUT/network/reflection.json" ||
    fail "network: slangc rejected the surface"
diff "$ROOT/bake/fixtures/network.reflection.json" "$OUT/network/reflection.json" ||
    fail "network: the reflection differs from the reference"

"$BAKE" libraries > "$OUT/libraries" || fail "libraries: the listing failed"
[ "$(head -1 "$OUT/libraries")" = "$(head -1 "$FIXTURES/network.manifest")" ] ||
    fail "libraries: the first line is not the manifest's MaterialX version"
grep -qE '^stdlib/stdlib_defs\.mtlx [0-9]+ [0-9]+$' "$OUT/libraries" ||
    fail "libraries: stdlib_defs.mtlx is not listed with its size and time"

sed 's/ND_geompropvalue_vector2/ND_missing/' "$ROOT/bake/fixtures/network.mtlx" > "$OUT/missing.mtlx"
if "$BAKE" generate "$OUT/missing.mtlx" "$OUT/missing" 2> "$OUT/missing.err"; then
    fail "missing: a document with an unknown nodedef generated"
fi
grep -q "ND_missing" "$OUT/missing.err" || fail "missing: the error does not name the nodedef"
# The vertices carry one float2 primvar; a surface reading a color3 one
# has no vertex stage.
sed -e 's|<input name="diffuseColor" nodename="n1" output="rgb" />|&\n    <input name="emissiveColor" nodename="n3" />|' \
    -e 's|  <surfacematerial|  <node name="n3" nodedef="ND_geompropvalue_color3">\n    <input name="geomprop" type="string" value="displayColor" />\n  </node>\n  <surfacematerial|' \
    "$ROOT/bake/fixtures/network.mtlx" > "$OUT/primvar.mtlx"
if "$BAKE" generate "$OUT/primvar.mtlx" "$OUT/primvar" 2> "$OUT/primvar.err"; then
    fail "primvar: a surface reading a color3 primvar generated"
fi
grep -q "i_geomprop_displayColor" "$OUT/primvar.err" || fail "primvar: the error does not name the varying"
# UV0, the texcoord the nodes read by default, is the primvar st: a texture
# whose st is unconnected reads the mesh's texcoord, and a surface reading
# UV0 beside another primvar has no vertex stage.
sed -e '/<input name="st" nodename="n2" \/>/d' "$ROOT/bake/fixtures/network.mtlx" > "$OUT/uv0.mtlx"
"$BAKE" generate "$OUT/uv0.mtlx" "$OUT/uv0" || fail "uv0: a surface reading UV0 failed to generate"
grep -q "o.texcoord_0 = p.texcoord;" "$OUT/uv0/surface.slang" ||
    fail "uv0: the vertex stage does not feed UV0 from the texcoord"
sed -e 's|value="st"|value="perfuv"|' \
    -e 's|<input name="diffuseColor" nodename="n1" output="rgb" />|&\n    <input name="emissiveColor" nodename="n3" output="rgb" />|' \
    -e 's|  <surfacematerial|  <node name="n3" nodedef="ND_UsdUVTexture_23" />\n  <surfacematerial|' \
    "$ROOT/bake/fixtures/network.mtlx" > "$OUT/perfuv.mtlx"
if "$BAKE" generate "$OUT/perfuv.mtlx" "$OUT/perfuv" 2> "$OUT/perfuv.err"; then
    fail "perfuv: a surface reading UV0 beside perfuv generated"
fi
grep -q "perfuv and st" "$OUT/perfuv.err" || fail "perfuv: the error does not name the primvars"

texture() {
    "$BAKE" texture "$2" "$3" "$4" "" "$OUT/textures/$5.texture" 0 "$FIXTURES/$1" || fail "$5: texture failed"
}
texture ramp.exr raw float 4096 ramp
cmp "$FIXTURES/ramp.texture" "$OUT/textures/ramp.texture" || fail "ramp: the bake differs from the reference"
texture swatch.png srgb_texture color 4096 swatch
texture swatch.png raw color 4096 swatch_raw
texture swatch.png raw single 4096 single
texture mips.exr raw data 4 mips
texture wrapped.exr raw data 4096 wrapped
texture mips.exr raw data 8 mips8
# "auto" is sRGB for an 8-bit image and raw for a float one.
texture swatch.png auto color 4096 swatch_auto
cmp "$OUT/textures/swatch.texture" "$OUT/textures/swatch_auto.texture" || fail "swatch_auto: auto is not sRGB for a PNG"
texture ramp.exr auto float 4096 ramp_auto
cmp "$FIXTURES/ramp.texture" "$OUT/textures/ramp_auto.texture" || fail "ramp_auto: auto is not raw for an EXR"
# A JPEG reads, and a space the config knows beyond sRGB is converted.
texture greys.jpg auto color 4096 greys
texture acescg.exr acescg color 4096 acescg
# A UDIM set, its tiles listed as the renderer lists them, is one array, a
# layer per tile, with a table from tile to layer; a tile numbered past the
# hundred is refused.
"$BAKE" texture raw float 4096 "" "$OUT/textures/udim.texture" 1001 "$FIXTURES/udim.1001.exr" 1012 "$FIXTURES/udim.1012.exr" ||
    fail "udim: texture failed"
"$BAKE" texture raw float 4096 "" "$OUT/textures/mixed.texture" 1001 "$FIXTURES/mixed.1001.exr" 1002 "$FIXTURES/mixed.1002.exr" ||
    fail "mixed: texture failed"
if "$BAKE" texture raw float 4096 "" "$OUT/textures/none.texture" 1101 "$FIXTURES/udim.1001.exr" 2> "$OUT/textures/none.err"; then
    fail "none: a tile numbered 1101 baked"
fi
grep -q "1101" "$OUT/textures/none.err" || fail "none: the error does not name the tile"
! cmp -s "$OUT/textures/swatch.texture" "$OUT/textures/swatch_raw.texture" || fail "swatch_raw: raw and sRGB bake alike, so the auto checks prove nothing"
"$HFS/bin/hython" "$ROOT/bake/texture.py" "$OUT/textures" || fail "textures: a bake does not decode to its source"
if "$BAKE" texture bogus color 4096 "" "$OUT/textures/bogus.texture" 0 "$FIXTURES/swatch.png" 2> "$OUT/textures/bogus.err"; then
    fail "bogus: a texture in an unknown colour space baked"
fi
grep -q "bogus" "$OUT/textures/bogus.err" || fail "bogus: the error does not name the colour space"

[ $failed = 0 ] && echo "PASS: bake"
exit $failed
