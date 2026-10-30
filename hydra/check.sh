#!/usr/bin/env bash
# The delegate's check, run on a GPU host with Houdini (not in CI).
#
# Builds the delegate and runs hydra/image.py, which renders each scene in
# hydra/images/ in husk (color as HdFormatFloat32Vec4, depth and normal) and
# in usdrecord (HdFormatFloat16Vec4 color), and compares the images with a ray
# cast from USD. usdrecord draws on a private Xvnc display, started here. The
# validation layer must be on and report nothing from Pika.
#
# Then it syncs each scene in hydra/fixtures/ for two frames, running the
# scene's .py script (if any) on husk's stage between them, and compares
# Pika's store dump with the scene's .dump, which was written by hand. Each
# run's PIKA_STATS file must summarise with hydra/stats.py, with every GPU
# pass timed and the driver's memory budget read.
#
# Last it renders hydra/scenes/cube.usda through UsdImagingGL in hython on
# the same display (hydra/engine.py), as usdview renders, and checks that the
# host sees Pika's stats through Hydra's render stats.
#
#   VULKAN_SDK=/path/to/vulkan_sdk/1.4.328.1/x86_64 hydra/check.sh
set -euo pipefail

HFS=${HFS:-/opt/hfs21.0}
SDK=${VULKAN_SDK:?set VULKAN_SDK to the SDK\'s x86_64 directory, for its validation layer}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/target/hydra/check

make -C "$ROOT/hydra" --no-print-directory
mkdir -p "$OUT/images" "$OUT/fixtures"

# houdini_setup_bash reads unset variables.
set +u
cd "$HFS" && source ./houdini_setup_bash >/dev/null && cd "$ROOT"
set -u
export PXR_PLUGINPATH_NAME=$ROOT/target/hydra
export LD_LIBRARY_PATH=$SDK/lib:${LD_LIBRARY_PATH:-}
export VK_ADD_LAYER_PATH=$SDK/share/vulkan/explicit_layer.d

failed=0
fail() {
    echo "FAIL: $*"
    failed=1
}

# clean <name> <log>: whether a husk or usdrecord log shows Pika validated and error free.
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
    PIKA_STORE_DUMP=$OUT/fixtures/$name.dump PIKA_STATS=$OUT/fixtures/$name.stats \
        husk -V 1 -R Pika --headlight none -f 1 -n 2 --res 16 16 \
        "${script[@]}" -o "$OUT/fixtures/$name.\$F4.exr" "$1" >"$OUT/fixtures/$name.log" 2>&1 ||
        fail "$name: husk exited $? (see $OUT/fixtures/$name.log)"
    clean "$name" "$OUT/fixtures/$name.log"
    # UsdImaging names native instance prototypes after hashes that change
    # from run to run.
    sed -i -E 's/(_purpose|_geomModel)[0-9a-f]+/\1#/g' "$OUT/fixtures/$name.dump"
    diff -u "$dir/$name.dump" <(awk '/^sync /{ on = ($2 <= 2) } on' "$OUT/fixtures/$name.dump") ||
        fail "$name: the store differs from $name.dump"
    # husk renders frame 2 again until Pika reports it delivered; those
    # renders' syncs must change nothing.
    local n
    for n in $(seq 3 "$(grep -c '^sync ' "$OUT/fixtures/$name.dump")"); do
        [ "$(store_after "$n" "$OUT/fixtures/$name.dump")" = "$(store_after 2 "$OUT/fixtures/$name.dump")" ] ||
            fail "$name: sync $n, a render repeated for frame 2, changed the store"
    done
    [ "$(store_after 2 "$OUT/fixtures/$name.dump")" != "$(store_after 1 "$dir/$name.dump")" ] ||
        fail "$name: frame 2's store matches frame 1's expectation"
    python3 -I "$ROOT/hydra/stats.py" "$OUT/fixtures/$name.stats" >"$OUT/fixtures/$name.summary" ||
        fail "$name: the stats do not summarise (see $OUT/fixtures/$name.summary)"
    grep -q "^  gpu_readback " "$OUT/fixtures/$name.summary" ||
        fail "$name: the summary has no GPU readback time, so a pass went untimed"
    grep -q "vram [1-9].* used of [1-9]" "$OUT/fixtures/$name.summary" ||
        fail "$name: the summary has no memory budget"
}

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
# Xvnc writes the display it chose to fd 3. It listens on no port, only on
# a socket in $OUT.
rm -f "$OUT/display" "$OUT/vnc"
Xvnc -displayfd 3 -rfbport -1 -rfbunixpath "$OUT/vnc" -nolisten tcp -SecurityTypes None \
    -geometry 640x480 -depth 24 3>"$OUT/display" 2>"$OUT/xvnc.log" &
xvnc=$!
trap 'kill $xvnc' EXIT
until [ -s "$OUT/display" ]; do
    kill -0 $xvnc 2>/dev/null || { echo "Xvnc did not start (see $OUT/xvnc.log)"; exit 1; }
    sleep 0.1
done
DISPLAY=:$(cat "$OUT/display") hython "$ROOT/hydra/image.py" "$OUT/images" usdrecord ||
    fail "the images differ from their references (see above)"
for log in "$OUT"/images/*.log; do
    clean "$(basename "$log" .log)" "$log"
done
fixtures=("$ROOT"/hydra/fixtures/*.usda)
for f in "${fixtures[@]}"; do
    fixture "$f"
done
echo "fixtures: checked ${#fixtures[@]} scenes"
# The stats reader's control: a first frame line cut short must fail it.
first=${fixtures[0]}
! python3 -I "$ROOT/hydra/stats.py" <(sed '0,/^frame /s/ latent [01]$//' "$OUT/fixtures/$(basename "$first" .usda).stats") \
    >/dev/null 2>&1 || fail "stats.py summarised a frame line cut short"

DISPLAY=:$(cat "$OUT/display") hython "$ROOT/hydra/engine.py" "$ROOT/hydra/scenes/cube.usda" >"$OUT/engine.log" 2>&1 ||
    fail "the host does not see Pika's stats (see $OUT/engine.log)"
clean engine "$OUT/engine.log"

[ $failed -eq 0 ] && echo "PASS" || exit 1
