#!/usr/bin/env bash
# The delegate's check, run on a GPU host with Houdini (not in CI).
#
# Builds the delegate for HFS into target/hydra/check/<houdini>/, where
# everything the run writes goes, so the two Houdinis' checks run side by
# side and neither touches target/hydra. Then, in order: hydra/image.py
# (husk and usdrecord renders against the ray cast), the store-dump
# fixtures in hydra/fixtures/ (two frames each, the scene's .py edit between
# them, RENDERS at a time; see fixture() for the header keys), and
# hydra/engine.py (UsdImagingGL in hython, as usdview renders). usdrecord
# and hython draw on a private Xvnc display started here. The validation
# layer must be on and report nothing from Pika.
#
# Materials need the bake tool, built against Houdini 22 whichever Houdini
# hosts the delegate (BAKE_HFS); the shaders it generates go to a cache of
# this run's own.
#
#   VULKAN_SDK=/path/to/vulkan_sdk/1.4.328.1/x86_64 hydra/check.sh
#   HFS=/opt/hfs22.0 VULKAN_SDK=... hydra/check.sh   # the other Houdini, at the same time
set -euo pipefail

HFS=${HFS:-/opt/hfs21.0}
BAKE_HFS=${BAKE_HFS:-/opt/hfs22.0}
SDK=${VULKAN_SDK:?set VULKAN_SDK to the SDK\'s x86_64 directory, for its validation layer}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/target/hydra/check/$(basename "$HFS")
RENDERS=8

make -C "$ROOT/hydra" --no-print-directory HFS="$HFS" OUT="$OUT/plugin"
# Both Houdinis' checks share the bake tool.
flock "$ROOT/target/bake.lock" make -C "$ROOT/bake" --no-print-directory HFS="$BAKE_HFS"
rm -rf "$OUT/cache" "$OUT/fixtures"
mkdir -p "$OUT/images" "$OUT/fixtures"
export PIKA_CACHE=$OUT/cache

# houdini_setup_bash reads unset variables.
set +u
cd "$HFS" && source ./houdini_setup_bash >/dev/null && cd "$ROOT"
set -u
export PXR_PLUGINPATH_NAME=$OUT/plugin
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

# counts <stats> <edit>: the surfaces compiled and textures baked for frame
# 1, then for frame 2: the frames from the edit's sync on.
counts() {
    awk -v edit="$2" '/^sync /{ f = ($2 < edit) ? 1 : 2; next }
        /^frame /{ for (i = 3; i < NF; i += 2) {
            if ($i == "compiled_surfaces") c[f] += $(i + 1)
            if ($i == "baked_textures") b[f] += $(i + 1) } }
        END { print c[1] + 0, b[1] + 0, c[2] + 0, b[2] + 0 }' "$1"
}

# expected <name> <count> <default>: what the scene's header expects of the
# count, in its "# <count>: ..." line, or the default.
expected() {
    local line
    line=$(sed -n "s/^# $2: //p" "$ROOT/hydra/fixtures/$1.usda" | head -n 1)
    echo "${line:-$3}"
}

# redundant <stats> <edit>: the mesh bytes the edit's sync handed over that
# the store held already.
redundant() {
    awk -v edit="$2" '$1 == "sync" && $2 == edit { for (i = 3; i < NF; i += 2)
        if ($i == "mesh_redundant_bytes") print $(i + 1) }' "$1"
}

# pending <stats>: the materials pending on the first frame and on the last.
pending() {
    awk '/^frame /{ for (i = 3; i < NF; i += 2) if ($i == "pending_materials") {
            if (!seen) first = $(i + 1); seen = 1; last = $(i + 1) } }
        END { print first + 0, last + 0 }' "$1"
}

# fixture <scene.usda>: frame 1 and the edit must each match their expected
# store. The edit must differ from frame 1's store, so a store left stale by
# the edit would fail.
fixture() {
    local name dir=$ROOT/hydra/fixtures script=()
    name=$(basename "$1" .usda)
    [ -f "$dir/$name.py" ] && script=(--postframe-script "$dir/$name.py")
    # A cache of its own, so the compile and bake counts below are its own
    # misses, not what the image scenes left behind.
    PIKA_CACHE=$OUT/fixtures/$name.cache PIKA_BUDGET=$(expected "$name" budget "") \
        PIKA_STORE_DUMP=$OUT/fixtures/$name.dump PIKA_STATS=$OUT/fixtures/$name.stats \
        husk -V 1 -R Pika --headlight none -f 1 -n 2 --res 16 16 \
        "${script[@]}" -o "$OUT/fixtures/$name.\$F4.exr" "$1" >"$OUT/fixtures/$name.log" 2>&1 ||
        fail "$name: husk exited $? (see $OUT/fixtures/$name.log)"
    clean "$name" "$OUT/fixtures/$name.log"
    # UsdImaging names native instance prototypes after hashes that change
    # from run to run and after the primvars it propagated, which changed
    # between USD 25 and 26; and assets resolve under this checkout.
    sed -i -E -e 's|/UsdNiPropagatedPrototypes/[^/]+/|/UsdNiPropagatedPrototypes/#/|g' \
        -e 's/ForInstancer[0-9a-f]+/ForInstancer#/g' -e "s|$ROOT/||g" "$OUT/fixtures/$name.dump"
    # husk syncs and renders a frame again until Pika reports it delivered,
    # which waits for its materials' builds, so the edit is the first sync
    # whose store differs from sync 1's; the syncs before and after it must
    # change nothing.
    local n edit= syncs
    syncs=$(grep -c '^sync ' "$OUT/fixtures/$name.dump")
    for n in $(seq 2 "$syncs"); do
        if [ "$(store_after "$n" "$OUT/fixtures/$name.dump")" != "$(store_after 1 "$OUT/fixtures/$name.dump")" ]; then
            edit=$n
            break
        fi
    done
    if [ -z "$edit" ]; then
        fail "$name: no sync changed the store, so frame 2 is frame 1"
        return
    fi
    diff -u "$dir/$name.dump" <(printf 'sync 1\n%s\nsync 2\n%s\n' \
        "$(store_after 1 "$OUT/fixtures/$name.dump")" "$(store_after "$edit" "$OUT/fixtures/$name.dump")") ||
        fail "$name: the store differs from $name.dump"
    for n in $(seq $((edit + 1)) "$syncs"); do
        [ "$(store_after "$n" "$OUT/fixtures/$name.dump")" = "$(store_after "$edit" "$OUT/fixtures/$name.dump")" ] ||
            fail "$name: sync $n, a render repeated for frame 2, changed the store"
    done
    python3 -I "$ROOT/hydra/stats.py" "$OUT/fixtures/$name.stats" >"$OUT/fixtures/$name.summary" ||
        fail "$name: the stats do not summarise (see $OUT/fixtures/$name.summary)"
    grep -q "^  gpu_readback " "$OUT/fixtures/$name.summary" ||
        fail "$name: the summary has no GPU readback time, so a pass went untimed"
    grep -q "vram [1-9].* used of [1-9]" "$OUT/fixtures/$name.summary" ||
        fail "$name: the summary has no memory budget"
    # An edit to a value, a wrap mode or an asset path among them, compiles
    # nothing; one to the network's structure, the primvar a reader names
    # included, compiles its surface; a texture is baked once per file. The
    # scene's header states what the run must count ("# key: value"; the
    # default, when absent, in brackets):
    #   compiles: A then B   surfaces compiled for frame 1, then for the edit (0 then 0)
    #   bakes: A then B      textures baked, likewise (0 then 0)
    #   warnings: N          material warnings, each once though husk renders a
    #                        frame more than once, and none for a repair (0)
    #   warned: M            one warning's text after "material ", which must
    #                        name what to fix and say what is drawn (unchecked)
    #   redundant bytes: R   mesh bytes the edit handed over that the store held;
    #                        0 for an edit touching no primvar the materials read
    #                        (unchecked: Hydra resends what a changed topology covers)
    #   budget: B            bytes standing in for the device's memory budget,
    #                        PIKA_BUDGET (none)
    # The scenes that edit a material are each other's control. Every material
    # with a surface in frame 1's expected store must be pending on the first
    # frame and none on the last; a scene without one is the control.
    local compiles bakes warnings warned redundant built c1 b1 c2 b2
    compiles=$(expected "$name" compiles "0 then 0")
    bakes=$(expected "$name" bakes "0 then 0")
    warnings=$(expected "$name" warnings 0)
    warned=$(expected "$name" warned "")
    redundant=$(expected "$name" "redundant bytes" "")
    built=$(awk '/^sync 2/{ exit } /^  surface / && $2 != "none" { n++ } END { print n + 0 }' "$dir/$name.dump")
    read -r c1 b1 c2 b2 <<<"$(counts "$OUT/fixtures/$name.stats" "$edit")"
    [ "$c1 then $c2" = "$compiles" ] ||
        fail "$name: compiled $c1 surfaces for frame 1 then $c2 for the edit, not $compiles"
    [ "$b1 then $b2" = "$bakes" ] ||
        fail "$name: baked $b1 textures for frame 1 then $b2 for the edit, not $bakes"
    [ "$(pending "$OUT/fixtures/$name.stats")" = "$built 0" ] ||
        fail "$name: $(pending "$OUT/fixtures/$name.stats") materials pending on the first and last frames, not $built 0"
    [ "$(grep -c '^pika: warning: material ' "$OUT/fixtures/$name.log")" = "$warnings" ] ||
        fail "$name: $(grep -c '^pika: warning: material ' "$OUT/fixtures/$name.log") material warnings, not $warnings"
    [ -z "$warned" ] || grep -qF "pika: warning: material $warned" "$OUT/fixtures/$name.log" ||
        fail "$name: no warning reads \"material $warned\" (see $OUT/fixtures/$name.log)"
    [ -z "$redundant" ] || [ "$(redundant "$OUT/fixtures/$name.stats" "$edit")" = "$redundant" ] ||
        fail "$name: the edit handed over $(redundant "$OUT/fixtures/$name.stats" "$edit") redundant mesh bytes, not $redundant"
}

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
# Xvnc writes the display it chose to fd 3. It listens on no port, only on
# a socket in a directory of this run's own under the runtime directory: a
# socket's path holds 108 bytes, which $OUT exceeds from a deep checkout.
RUN=$(mktemp -d "${XDG_RUNTIME_DIR:-/tmp}/pika-check.XXXXXX")
rm -f "$OUT/display"
Xvnc -displayfd 3 -rfbport -1 -rfbunixpath "$RUN/vnc" -nolisten tcp -SecurityTypes None \
    -geometry 640x480 -depth 24 3>"$OUT/display" 2>"$OUT/xvnc.log" &
xvnc=$!
trap 'kill $xvnc; rm -rf "$RUN"' EXIT
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
# Each fixture's check reports to a file of its own; one that stops short of
# the end exits nonzero.
export -f fixture clean store_after counts expected pending redundant fail
export OUT ROOT
printf '%s\n' "${fixtures[@]}" | xargs -P "$RENDERS" -I{} bash -c \
    'set -euo pipefail; fixture "$1" >"$OUT/fixtures/$(basename "$1" .usda).report" 2>&1' _ {} ||
    fail "a fixture's check stopped short (see $OUT/fixtures/*.report)"
for f in "${fixtures[@]}"; do
    cat "$OUT/fixtures/$(basename "$f" .usda).report"
    grep -q "^FAIL: " "$OUT/fixtures/$(basename "$f" .usda).report" && failed=1
done
echo "fixtures: checked ${#fixtures[@]} scenes"
# The stats reader's control: a first frame line cut short must fail it.
first=${fixtures[0]}
! python3 -I "$ROOT/hydra/stats.py" <(sed '0,/^frame /s/ [a-z_]* [0-9.]*$//' "$OUT/fixtures/$(basename "$first" .usda).stats") \
    >/dev/null 2>&1 || fail "stats.py summarised a frame line cut short"

DISPLAY=:$(cat "$OUT/display") hython "$ROOT/hydra/engine.py" "$ROOT/hydra/scenes/cube.usda" \
    "$ROOT/hydra/scenes/painted.usda" >"$OUT/engine.log" 2>&1 ||
    fail "the host does not see Pika's stats, or a pending material is not clay or as it was (see $OUT/engine.log)"
clean engine "$OUT/engine.log"

[ $failed -eq 0 ] && echo "PASS" || exit 1
