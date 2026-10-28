#!/usr/bin/env bash
# The delegate skeleton's check, run on a GPU host with Houdini (not in CI).
#
# Builds the delegate and renders hydra/scenes/cube.usda in husk, twice: with
# husk's default product, which binds color as HdFormatFloat32Vec4, and with
# the scene's /render/half settings, which bind HdFormatFloat16Vec4. Every
# frame must be exactly Pika's clear color, and must differ from the host's
# clear value (0, 0, 0, 0), which is what husk saves when Pika writes nothing.
# The validation layer must be on and report nothing from Pika.
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
mkdir -p "$OUT/float32" "$OUT/float16"

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

# render <name> [husk options]: renders FRAMES frames into $OUT/<name>/.
render() {
    local name=$1
    shift
    husk -V 1 -R Pika --disable-motionblur -f 1 -n $FRAMES --res 1920 1080 "$@" \
        -o "$OUT/$name/cube.\$F4.exr" hydra/scenes/cube.usda >"$OUT/$name.log" 2>&1 ||
        fail "$name: husk exited $? (see $OUT/$name.log)"

    grep -q "^pika: rendering on .*validation on (sync)" "$OUT/$name.log" ||
        fail "$name: Pika did not open a device with validation on"
    grep "^pika: error" "$OUT/$name.log" && fail "$name: Pika reported errors"
    # husk itself loads Houdini's SwiftShader driver through a symlinked path,
    # which the loader reports to every Vulkan instance in the process.
    grep "^validation:" "$OUT/$name.log" | grep -v "libvk_swiftshader.so was found to differ" &&
        fail "$name: the validation layer reported messages"

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

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
render float32 -c /world/camera
render float16 -s /render/half

[ $failed -eq 0 ] && echo "PASS" || exit 1
