#!/usr/bin/env bash
# The bake tools' check, run where Houdini 22 and slangc are (not in CI).
#
# Builds the tools, and generates bake/fixtures/network.mtlx, the document
# src/material.rs writes for hydra/fixtures/network.usda. The manifest must
# match bake/fixtures/network.manifest, the pixel stage must read the eye
# and the lights from the view (shaders/frame.slang) rather than declare
# them, run Pika's light loop with none of MaterialX's light code, and the
# surface must compile as Pika compiles it. A document naming
# a nodedef the libraries lack, or reading a primvar the renderer's vertices
# lack, or UV0 beside another primvar, or a uniform the view does not
# provide, must fail, naming it. A surface reading UV0 alone reads the
# texcoord. generate libraries must list the manifest's MaterialX version
# and the standard library's definitions with their size and time, and
# texture libraries the OpenImageIO and OpenColorIO versions beneath the
# texture tool.
#
#   bake/check.sh
set -euo pipefail

HFS=${HFS:-/opt/hfs22.0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/target/bake/check
GENERATE=$ROOT/target/bake/generate
TEXTURE=$ROOT/target/bake/texture

make -C "$ROOT/bake" --no-print-directory HFS="$HFS"
rm -rf "$OUT"
mkdir -p "$OUT/network" "$OUT/missing" "$OUT/primvar" "$OUT/uv0" "$OUT/perfuv" "$OUT/time" \
    "$OUT/textures"
FIXTURES=$ROOT/bake/fixtures

failed=0
fail() {
    echo "FAIL: $*"
    failed=1
}

"$GENERATE" "$ROOT/bake/fixtures/network.mtlx" "$OUT/network" ||
    fail "network: generate failed"
diff "$ROOT/bake/fixtures/network.manifest" "$OUT/network/manifest" ||
    fail "network: the manifest differs from the reference"
grep -qx '#include "frame.slang"' "$OUT/network/pixel.slang" ||
    fail "network: the pixel stage does not read the frame through shaders/frame.slang"
! grep -q "u_viewPosition;" "$OUT/network/pixel.slang" ||
    fail "network: the pixel stage declares the eye in the material's block"
grep -q "while (lights.next(light_index))" "$OUT/network/pixel.slang" ||
    fail "network: the pixel stage does not run Pika's light loop"
! grep -qE "LightData|sampleLightSource|numActiveLightSources" "$OUT/network/pixel.slang" ||
    fail "network: the pixel stage holds MaterialX's light code"
# The flags are the renderer's (src/shader.rs: FLAGS and SURFACE_FLAGS). The
# reflection must match bake/fixtures/network.reflection.json, the layout
# test's input, as JSON: copy it over when slangc or the libraries change.
slangc "$OUT/network/surface.slang" -I "$ROOT/shaders" -target spirv -fvk-use-entrypoint-name \
    -warnings-as-errors all -warnings-disable 31106,31107 -o "$OUT/network/surface.spv" \
    -reflection-json "$OUT/network/reflection.json" ||
    fail "network: slangc rejected the surface"
python3 -I -c 'import json, sys; sys.exit(json.load(open(sys.argv[1])) != json.load(open(sys.argv[2])))' \
    "$FIXTURES/network.reflection.json" "$OUT/network/reflection.json" ||
    fail "network: the reflection differs from the reference ($OUT/network/reflection.json is this run's)"

"$GENERATE" libraries > "$OUT/libraries" || fail "libraries: the listing failed"
[ "$(head -1 "$OUT/libraries")" = "$(head -1 "$FIXTURES/network.manifest")" ] ||
    fail "libraries: the first line is not the manifest's MaterialX version"
grep -qE '^stdlib/stdlib_defs\.mtlx [0-9]+ [0-9]+$' "$OUT/libraries" ||
    fail "libraries: stdlib_defs.mtlx is not listed with its size and time"
"$TEXTURE" libraries > "$OUT/texture_libraries" || fail "texture libraries: the listing failed"
grep -qE '^openimageio [0-9.]+$' "$OUT/texture_libraries" ||
    fail "texture libraries: OpenImageIO's version is not listed"
grep -qE '^opencolorio [0-9.]+$' "$OUT/texture_libraries" ||
    fail "texture libraries: OpenColorIO's version is not listed"

sed 's/ND_geompropvalue_vector2/ND_missing/' "$ROOT/bake/fixtures/network.mtlx" > "$OUT/missing.mtlx"
if "$GENERATE" "$OUT/missing.mtlx" "$OUT/missing" 2> "$OUT/missing.err"; then
    fail "missing: a document with an unknown nodedef generated"
fi
grep -q "ND_missing" "$OUT/missing.err" || fail "missing: the error does not name the nodedef"
# The vertices carry one float2 primvar; a surface reading a color3 one
# has no vertex stage.
sed -e 's|<input name="diffuseColor" nodename="n1" output="rgb" />|&\n    <input name="emissiveColor" nodename="n3" />|' \
    -e 's|  <surfacematerial|  <node name="n3" nodedef="ND_geompropvalue_color3">\n    <input name="geomprop" type="string" value="displayColor" />\n  </node>\n  <surfacematerial|' \
    "$ROOT/bake/fixtures/network.mtlx" > "$OUT/primvar.mtlx"
if "$GENERATE" "$OUT/primvar.mtlx" "$OUT/primvar" 2> "$OUT/primvar.err"; then
    fail "primvar: a surface reading a color3 primvar generated"
fi
grep -q "i_geomprop_displayColor" "$OUT/primvar.err" || fail "primvar: the error does not name the varying"
# UV0 is st; UV0 beside another primvar has no vertex stage.
sed -e '/<input name="st" nodename="n2" \/>/d' "$ROOT/bake/fixtures/network.mtlx" > "$OUT/uv0.mtlx"
"$GENERATE" "$OUT/uv0.mtlx" "$OUT/uv0" || fail "uv0: a surface reading UV0 failed to generate"
grep -q "o.texcoord_0 = p.texcoord;" "$OUT/uv0/surface.slang" ||
    fail "uv0: the vertex stage does not feed UV0 from the texcoord"
sed -e 's|value="st"|value="perfuv"|' \
    -e 's|<input name="diffuseColor" nodename="n1" output="rgb" />|&\n    <input name="emissiveColor" nodename="n3" output="rgb" />|' \
    -e 's|  <surfacematerial|  <node name="n3" nodedef="ND_UsdUVTexture_23" />\n  <surfacematerial|' \
    "$ROOT/bake/fixtures/network.mtlx" > "$OUT/perfuv.mtlx"
if "$GENERATE" "$OUT/perfuv.mtlx" "$OUT/perfuv" 2> "$OUT/perfuv.err"; then
    fail "perfuv: a surface reading UV0 beside perfuv generated"
fi
grep -q "perfuv and st" "$OUT/perfuv.err" || fail "perfuv: the error does not name the primvars"
# A time node asks the generator for u_time, which the view does not hold.
sed -e 's|<input name="diffuseColor" nodename="n1" output="rgb" />|&\n    <input name="roughness" nodename="n3" />|' \
    -e 's|  <surfacematerial|  <node name="n3" nodedef="ND_time_float" />\n  <surfacematerial|' \
    "$ROOT/bake/fixtures/network.mtlx" > "$OUT/time.mtlx"
if "$GENERATE" "$OUT/time.mtlx" "$OUT/time" 2> "$OUT/time.err"; then
    fail "time: a surface reading u_time generated"
fi
grep -q "u_time" "$OUT/time.err" || fail "time: the error does not name the uniform"

texture() {
    "$TEXTURE" "$2" "$3" "" "$OUT/textures/$4.texture" 0 "$FIXTURES/$1" || fail "$4: texture failed"
}
texture ramp.exr raw float ramp
cmp "$FIXTURES/ramp.texture" "$OUT/textures/ramp.texture" || fail "ramp: the bake differs from the reference"
texture swatch.png srgb_texture color swatch
texture swatch.png raw color swatch_raw
texture swatch.png raw single single
texture mips.exr raw data mips
texture wrapped.exr raw data wrapped
# "auto" is sRGB for an 8-bit image and raw for a float one.
texture swatch.png auto color swatch_auto
cmp "$OUT/textures/swatch.texture" "$OUT/textures/swatch_auto.texture" || fail "swatch_auto: auto is not sRGB for a PNG"
texture ramp.exr auto float ramp_auto
cmp "$FIXTURES/ramp.texture" "$OUT/textures/ramp_auto.texture" || fail "ramp_auto: auto is not raw for an EXR"
# A JPEG reads, and a space the config knows beyond sRGB is converted.
texture greys.jpg auto color greys
texture acescg.exr acescg color acescg
# A UDIM set, its tiles listed as the renderer lists them, is one array, a
# layer per tile, with a table from tile to layer; a tile numbered past the
# hundred is refused.
"$TEXTURE" raw float "" "$OUT/textures/udim.texture" 1001 "$FIXTURES/udim.1001.exr" 1012 "$FIXTURES/udim.1012.exr" ||
    fail "udim: texture failed"
"$TEXTURE" raw float "" "$OUT/textures/mixed.texture" 1001 "$FIXTURES/mixed.1001.exr" 1002 "$FIXTURES/mixed.1002.exr" ||
    fail "mixed: texture failed"
if "$TEXTURE" raw float "" "$OUT/textures/none.texture" 1101 "$FIXTURES/udim.1001.exr" 2> "$OUT/textures/none.err"; then
    fail "none: a tile numbered 1101 baked"
fi
grep -q "1101" "$OUT/textures/none.err" || fail "none: the error does not name the tile"
! cmp -s "$OUT/textures/swatch.texture" "$OUT/textures/swatch_raw.texture" || fail "swatch_raw: raw and sRGB bake alike, so the auto checks prove nothing"
"$HFS/bin/hython" "$ROOT/bake/texture.py" "$OUT/textures" || fail "textures: a bake does not decode to its source"
if "$TEXTURE" bogus color "" "$OUT/textures/bogus.texture" 0 "$FIXTURES/swatch.png" 2> "$OUT/textures/bogus.err"; then
    fail "bogus: a texture in an unknown colour space baked"
fi
grep -q "bogus" "$OUT/textures/bogus.err" || fail "bogus: the error does not name the colour space"

[ $failed = 0 ] && echo "PASS: bake"
exit $failed
