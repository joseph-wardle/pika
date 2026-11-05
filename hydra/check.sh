#!/usr/bin/env bash
# The delegate's check, run on a GPU host with Houdini (not in CI).
#
# Builds the delegate for HFS into a directory of its own under
# target/hydra/check, named after the Houdini, where everything else the run
# writes goes too, so the check of one Houdini runs beside the other's and
# neither touches the delegate target/hydra holds for usdview. Then it runs
# hydra/image.py, which renders each scene in hydra/images/ in husk (color as
# HdFormatFloat32Vec4, depth and normal) and in usdrecord (HdFormatFloat16Vec4
# color), and compares the images with a ray cast from USD. usdrecord draws on
# a private Xvnc display, started here. The validation layer must be on and
# report nothing from Pika.
#
# Then it syncs each scene in hydra/fixtures/ for two frames, RENDERS at a
# time as hydra/image.py renders, running the
# scene's .py script (if any) on husk's stage between them, and compares
# Pika's store dump with the scene's .dump, which was written by hand. Each
# run's PIKA_STATS file must summarise with hydra/stats.py, with every GPU
# pass timed and the driver's memory budget read, frame 2 must have
# compiled a surface and baked a texture only where the edit calls for one,
# a material must be pending while its build is out and not after, and a
# material that cannot be built must be warned of once.
#
# Last it renders hydra/scenes/cube.usda and painted.usda through
# UsdImagingGL in hython on the same display (hydra/engine.py), as usdview
# renders, and checks that the host sees Pika's stats through Hydra's render
# stats, clay where a material is first being built and the material as it
# was while an edit's build is out.
#
# Materials need the bake tool, built against Houdini 22 whichever Houdini
# hosts the delegate (BAKE_HFS), and the shaders it generates go to a cache
# of this run's own.
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
    PIKA_CACHE=$OUT/fixtures/$name.cache \
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
    # An edit to a value, a wrap mode among them, or an asset path compiles
    # nothing; one to the network's structure, the primvar a reader names
    # included, compiles its surface; a texture is baked once per file. The
    # scene's header states its counts: "# compiles: A then B" and
    # "# bakes: A then B" are the surfaces compiled and textures baked for
    # frame 1 and for the edit, 0 when absent, and "# warnings: N" the
    # warnings of materials that cannot be built, each warned of once when
    # its build lands, though husk renders a frame more than once, and
    # compiled again by its repair with no second warning. The scenes that
    # edit a material are each other's control, and each must compile within
    # its frame, so a counter that never moves fails. Every material with a
    # surface in frame 1's expected store is built off the render thread and
    # pending on the first frame, built or failed, and none is pending on the
    # last; a scene without one is the control. An edit that touches a
    # primvar the materials do not read hands over no mesh bytes the store
    # holds already ("# redundant bytes: 0"); the other edits are not held to
    # a count, since Hydra resends what a changed topology or instancing
    # covers.
    local compiles bakes warnings redundant built c1 b1 c2 b2
    compiles=$(expected "$name" compiles "0 then 0")
    bakes=$(expected "$name" bakes "0 then 0")
    warnings=$(expected "$name" warnings 0)
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
