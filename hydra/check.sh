#!/usr/bin/env bash
# The delegate's check, run on a GPU host with Houdini (not in CI).
#
# Builds the delegate and renders hydra/scenes/cube.usda in husk, twice: with
# husk's default product, which binds color as HdFormatFloat32Vec4, and with
# the scene's /render/half settings, which bind HdFormatFloat16Vec4. Every
# frame must be exactly Pika's clear color, and must differ from the host's
# clear value (0, 0, 0, 0), which is what husk saves when Pika writes nothing.
# The validation layer must be on and report nothing from Pika.
#
# Then it syncs each scene in hydra/fixtures/ for two frames, running the
# scene's .py script (if any) on husk's stage between them, and compares
# Pika's store dump with the scene's .dump, which was written by hand.
#
#   VULKAN_SDK=/path/to/vulkan_sdk/1.4.328.1/x86_64 hydra/check.sh
set -euo pipefail

HFS=${HFS:-/opt/hfs21.0}
SDK=${VULKAN_SDK:?set VULKAN_SDK to the SDK\'s x86_64 directory, for its validation layer}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/target/hydra/check
FRAMES=24
# src/renderer.rs CLEAR.
CLEAR=0.25,0.5,0.75,1

make -C "$ROOT/hydra" --no-print-directory
mkdir -p "$OUT/float32" "$OUT/float16" "$OUT/fixtures"

# houdini_setup_bash reads unset variables.
set +u
cd "$HFS" && source ./houdini_setup_bash >/dev/null && cd "$ROOT"
set -u
export PXR_PLUGINPATH_NAME=$ROOT/target/hydra
export LD_LIBRARY_PATH=$SDK/lib:${LD_LIBRARY_PATH:-}
export VK_ADD_LAYER_PATH=$SDK/share/vulkan/explicit_layer.d

hoiiotool --pattern constant:color=$CLEAR 1920x1080 4 -d half -o "$OUT/expected.exr"
hoiiotool --pattern constant:color=0,0,0,0 1920x1080 4 -d half -o "$OUT/unwritten.exr"

failed=0
fail() {
    echo "FAIL: $*"
    failed=1
}

# same <a> <b>: whether two images are identical. Houdini's idiff is not
# OIIO's, so this uses hoiiotool, which fails on a one-ULP half difference.
same() {
    hoiiotool --fail 0 --warn 0 "$1" "$2" --diff >/dev/null
}

# clean <name> <log>: whether a husk log shows Pika validated and error free.
clean() {
    grep -q "^pika: rendering on .*validation on (sync)" "$2" ||
        fail "$1: Pika did not open a device with validation on"
    grep "^pika: error" "$2" && fail "$1: Pika reported errors"
    # husk itself loads Houdini's SwiftShader driver through a symlinked path,
    # which the loader reports to every Vulkan instance in the process.
    grep "^validation:" "$2" | grep -v "libvk_swiftshader.so was found to differ" &&
        fail "$1: the validation layer reported messages"
    return 0
}

# render <name> [husk options]: renders FRAMES frames into $OUT/<name>/.
render() {
    local name=$1
    shift
    husk -V 1 -R Pika --disable-motionblur -f 1 -n $FRAMES --res 1920 1080 "$@" \
        -o "$OUT/$name/cube.\$F4.exr" hydra/scenes/cube.usda >"$OUT/$name.log" 2>&1 ||
        fail "$name: husk exited $? (see $OUT/$name.log)"

    clean "$name" "$OUT/$name.log"

    local f exact=0 control=0
    for f in "$OUT/$name"/cube.*.exr; do
        same "$OUT/expected.exr" "$f" && exact=$((exact + 1))
        same "$OUT/unwritten.exr" "$f" || control=$((control + 1))
    done
    echo "$name: $exact of $FRAMES frames are exactly the clear color; $control differ from unwritten"
    [ $exact -eq $FRAMES ] || fail "$name: frames differ from the clear color"
    [ $control -eq $FRAMES ] || fail "$name: frames match an unwritten buffer"
    grep "^pika: .* frames, last at" "$OUT/$name.log" || fail "$name: no frame timings"
}

# store_after <n> <dump>: the store as sync n left it.
store_after() {
    awk -v header="sync $1" '/^sync /{ on = ($0 == header); next } on' "$2"
}

# fixture <scene.usda>: frame 1 and frame 2 must each match their expected
# store. Frame 2 must also differ from frame 1's expectation, so a store left
# stale by the edit would fail.
fixture() {
    local name dir=$ROOT/hydra/fixtures script=()
    name=$(basename "$1" .usda)
    [ -f "$dir/$name.py" ] && script=(--postframe-script "$dir/$name.py")
    PIKA_STORE_DUMP=$OUT/fixtures/$name.dump husk -V 1 -R Pika --headlight none -f 1 -n 2 --res 16 16 \
        "${script[@]}" -o "$OUT/fixtures/$name.\$F4.exr" "$1" >"$OUT/fixtures/$name.log" 2>&1 ||
        fail "$name: husk exited $? (see $OUT/fixtures/$name.log)"
    clean "$name" "$OUT/fixtures/$name.log"
    # UsdImaging names native instance prototypes after hashes that change
    # from run to run.
    sed -i -E 's/(_purpose|_geomModel)[0-9a-f]+/\1#/g' "$OUT/fixtures/$name.dump"
    diff -u "$dir/$name.dump" "$OUT/fixtures/$name.dump" ||
        fail "$name: the store differs from $name.dump"
    [ "$(store_after 2 "$OUT/fixtures/$name.dump")" != "$(store_after 1 "$dir/$name.dump")" ] ||
        fail "$name: frame 2's store matches frame 1's expectation"
}

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
render float32 -c /world/camera
render float16 -s /render/half
fixtures=("$ROOT"/hydra/fixtures/*.usda)
for f in "${fixtures[@]}"; do
    fixture "$f"
done
echo "fixtures: checked ${#fixtures[@]} scenes"

[ $failed -eq 0 ] && echo "PASS" || exit 1
