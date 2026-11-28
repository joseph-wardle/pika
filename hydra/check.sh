#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RENDERS=8
if [ -n "${HFS:-}" ]; then
    BAKE_HFS=${BAKE_HFS:-$HFS}
    OUT=$ROOT/target/hydra/check/$(basename "$HFS")
    PYTHON=hython
    make -C "$ROOT/hydra" --no-print-directory HFS="$HFS" OUT="$OUT/plugin"
    flock "$ROOT/target/bake.lock" make -C "$ROOT/bake" --no-print-directory HFS="$BAKE_HFS"
    # Houdini must not load the shell's Python packages or Qt plugins.
    unset PYTHONPATH QT_PLUGIN_PATH
    LOADER=${LD_LIBRARY_PATH:-}
    # houdini_setup_bash reads unset variables.
    set +u
    cd "$HFS" && source ./houdini_setup_bash >/dev/null && cd "$ROOT"
    set -u
    # Houdini's Vulkan loader is too old for the validation layer's settings.
    export LD_LIBRARY_PATH=$LOADER:${LD_LIBRARY_PATH:-}
    export MATERIALX=$HFS/houdini/materialx
else
    OUT=$ROOT/target/hydra/check/usd
    PYTHON=python3
    make -C "$ROOT/hydra" --no-print-directory OUT="$OUT/plugin"
    flock "$ROOT/target/bake.lock" make -C "$ROOT/bake" --no-print-directory
fi
rm -rf "$OUT/cache" "$OUT/fixtures"
mkdir -p "$OUT/images" "$OUT/fixtures"
export PIKA_CACHE=$OUT/cache
export PXR_PLUGINPATH_NAME=$OUT/plugin
if [ -n "${VULKAN_SDK:-}" ]; then
    export LD_LIBRARY_PATH=$VULKAN_SDK/lib:${LD_LIBRARY_PATH:-}
    export VK_ADD_LAYER_PATH=$VULKAN_SDK/share/vulkan/explicit_layer.d
fi

failed=0
fail() {
    echo "FAIL: $*"
    failed=1
}

clean() {
    grep -q "^pika: rendering on .*validation on (sync)" "$2" ||
        fail "$1: Pika did not open a device with validation on"
    grep "^pika: error" "$2" && fail "$1: Pika reported errors"
    # husk ships its own SwiftShader driver.
    grep "^validation:" "$2" | grep -v "libvk_swiftshader.so was found to differ" &&
        fail "$1: the validation layer reported messages"
    return 0
}

store_after() {
    awk -v header="sync $1" '/^sync /{ on = ($0 == header); next } on' "$2"
}

# The first sync whose store differs from sync 1's, then any later one that differs from it.
edit_of() {
    awk '/^sync /{ n = $2; next } { store[n] = store[n] $0 "\n" }
        END { for (i = 2; i <= n; i++) {
            if (!edit && store[i] != store[1]) { edit = i; print i }
            else if (edit && store[i] != store[edit]) print i } }' "$1"
}

# Surfaces compiled and textures baked for frame 1, then from the edit on.
counts() {
    awk -v edit="$2" '/^sync /{ f = ($2 < edit) ? 1 : 2; next }
        /^frame /{ for (i = 3; i < NF; i += 2) {
            if ($i == "compiled_surfaces") c[f] += $(i + 1)
            if ($i == "baked_textures") b[f] += $(i + 1) } }
        END { print c[1] + 0, b[1] + 0, c[2] + 0, b[2] + 0 }' "$1"
}

# The scene header's "# key: value" lines.
keys() {
    awk '/^#$/{ n = 0; after = 1; next } /^#/{ if (after) line[++n] = $0; next }
        /^def /{ for (i = 1; i <= n; i++) print line[i]; exit }' "$ROOT/hydra/fixtures/$1.usda"
}

expected() {
    local line
    line=$(keys "$1" | sed -n "s/^# $2: //p" | head -n 1)
    echo "${line:-$3}"
}

redundant() {
    awk -v edit="$2" '$1 == "sync" && $2 == edit { for (i = 3; i < NF; i += 2)
        if ($i == "mesh_redundant_bytes") print $(i + 1) }' "$1"
}

pending() {
    awk '/^frame /{ for (i = 3; i < NF; i += 2) if ($i == "pending_materials") {
            if (!seen) first = $(i + 1); seen = 1; last = $(i + 1) } }
        END { print first + 0, last + 0 }' "$1"
}

two_frames() {
    local name
    name=$(basename "$1" .usda)
    if [ -n "${HFS:-}" ]; then
        husk -V 1 -R Pika --headlight none -f 1 -n 2 --res 16 16 ${2:+--postframe-script "$2"} \
            -o "$OUT/fixtures/$name.\$F4.exr" "$1"
    else
        DISPLAY=:$(cat "$OUT/display") "$PYTHON" "$ROOT/hydra/store.py" "$@"
    fi
}

fixture() {
    local name dir=$ROOT/hydra/fixtures script=()
    name=$(basename "$1" .usda)
    [ -f "$dir/$name.py" ] && script=("$dir/$name.py")
    PIKA_CACHE=$OUT/fixtures/$name.cache PIKA_BUDGET=$(expected "$name" budget "") \
        PIKA_STORE_DUMP=$OUT/fixtures/$name.dump PIKA_STATS=$OUT/fixtures/$name.stats \
        two_frames "$1" "${script[@]}" >"$OUT/fixtures/$name.log" 2>&1 ||
        fail "$name: the host exited $? (see $OUT/fixtures/$name.log)"
    clean "$name" "$OUT/fixtures/$name.log"
    # Prototype names hold per-run hashes, and asset paths this checkout.
    sed -i -E -e 's|/UsdNiPropagatedPrototypes/[^/]+/|/UsdNiPropagatedPrototypes/#/|g' \
        -e 's/ForInstancer[0-9a-f]+/ForInstancer#/g' -e "s|$ROOT/||g" "$OUT/fixtures/$name.dump"
    local n edit= changed
    changed=$(edit_of "$OUT/fixtures/$name.dump")
    edit=$(head -n 1 <<<"$changed")
    if [ -z "$edit" ]; then
        fail "$name: no sync changed the store, so frame 2 is frame 1"
        return
    fi
    printf 'sync 1\n%s\nsync 2\n%s\n' "$(store_after 1 "$OUT/fixtures/$name.dump")" \
        "$(store_after "$edit" "$OUT/fixtures/$name.dump")" >"$OUT/fixtures/$name.store"
    diff -u "$dir/$name.dump" "$OUT/fixtures/$name.store" ||
        fail "$name: the store differs from $name.dump ($OUT/fixtures/$name.store is this run's)"
    for n in $(tail -n +2 <<<"$changed"); do
        fail "$name: sync $n, a render repeated for frame 2, changed the store"
    done
    python3 -I "$ROOT/hydra/stats.py" "$OUT/fixtures/$name.stats" >"$OUT/fixtures/$name.summary" ||
        fail "$name: the stats do not summarise (see $OUT/fixtures/$name.summary)"
    grep -q "^  gpu_readback " "$OUT/fixtures/$name.summary" ||
        fail "$name: the summary has no GPU readback time, so a pass went untimed"
    grep -q "vram [1-9].* used of [1-9]" "$OUT/fixtures/$name.summary" ||
        fail "$name: the summary has no memory budget"
    local key
    while IFS= read -r key; do
        case $key in
            "# compiles: "* | "# bakes: "* | "# warnings: "* | "# warned: "* | "# redundant bytes: "* | "# budget: "*) ;;
            *) fail "$name: the header line \"$key\" is not a key this check reads" ;;
        esac
    done < <(keys "$name")
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
    [ -z "$warned" ] || grep -qF "pika: warning: $warned" "$OUT/fixtures/$name.log" ||
        fail "$name: no warning reads \"$warned\" (see $OUT/fixtures/$name.log)"
    [ -z "$redundant" ] || [ "$(redundant "$OUT/fixtures/$name.stats" "$edit")" = "$redundant" ] ||
        fail "$name: the edit handed over $(redundant "$OUT/fixtures/$name.stats" "$edit") redundant mesh bytes, not $redundant"
}

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
rm -f "$OUT/display"
Xvfb -displayfd 3 -nolisten tcp -screen 0 640x480x24 3>"$OUT/display" 2>"$OUT/xvfb.log" &
xvfb=$!
trap 'kill $xvfb' EXIT
until [ -s "$OUT/display" ]; do
    kill -0 $xvfb 2>/dev/null || { echo "Xvfb did not start (see $OUT/xvfb.log)"; exit 1; }
    sleep 0.1
done
DISPLAY=:$(cat "$OUT/display") "$PYTHON" "$ROOT/hydra/image.py" "$OUT/images" usdrecord ||
    fail "the images differ from their references (see above)"
for log in "$OUT"/images/*.log; do
    clean "$(basename "$log" .log)" "$log"
done
# An empty shadow pool is refused. Only husk hands Pika the render settings' pool.
if [ -n "${HFS:-}" ]; then
cat >"$OUT/unpooled.usda" <<EOF
#usda 1.0
(
    subLayers = [@$ROOT/hydra/images/receding.usda@]
)

over "render"
{
    over "settings"
    {
        custom int pika:shadowPool = 0
    }
}
EOF
husk -V 1 -R Pika -s /render/settings -f 1 -n 1 -o "$OUT/unpooled.\$F4.exr" "$OUT/unpooled.usda" \
    >"$OUT/unpooled.log" 2>&1 || true
grep -q "^pika: error: render: pika:shadowPool is 0 MB" "$OUT/unpooled.log" ||
    fail "an empty shadow pool was not refused (see $OUT/unpooled.log)"
fi
fixtures=("$ROOT"/hydra/fixtures/*.usda)
export -f fixture two_frames clean store_after edit_of counts keys expected pending redundant fail
export OUT ROOT PYTHON
printf '%s\n' "${fixtures[@]}" | xargs -P "$RENDERS" -I{} bash -c \
    'set -euo pipefail; fixture "$1" >"$OUT/fixtures/$(basename "$1" .usda).report" 2>&1' _ {} ||
    fail "a fixture's check stopped short (see $OUT/fixtures/*.report)"
for f in "${fixtures[@]}"; do
    cat "$OUT/fixtures/$(basename "$f" .usda).report"
    grep -q "^FAIL: " "$OUT/fixtures/$(basename "$f" .usda).report" && failed=1
done
echo "fixtures: checked ${#fixtures[@]} scenes"
# Control: a frame line cut short must fail the stats reader.
first=${fixtures[0]}
! python3 -I "$ROOT/hydra/stats.py" <(sed '0,/^frame /s/ [a-z_]* [0-9.]*$//' "$OUT/fixtures/$(basename "$first" .usda).stats") \
    >/dev/null 2>&1 || fail "stats.py summarised a frame line cut short"

DISPLAY=:$(cat "$OUT/display") "$PYTHON" "$ROOT/hydra/engine.py" "$ROOT/hydra/scenes/cube.usda" \
    "$ROOT/hydra/scenes/painted.usda" "$ROOT/hydra/scenes/mipped.usda" "$ROOT/hydra/scenes/penumbra.usda" \
    >"$OUT/engine.log" 2>&1 ||
    fail "the host does not see Pika's stats, a pending material is not clay or as it was, a cap change in session differs from a fresh session, or a soft shadow does not accumulate (see $OUT/engine.log)"
clean engine "$OUT/engine.log"
grep -q "^pika: warning: texture .*mips.exr.*cannot read its levels.*; held at 2×2 until the memory budget grows or the textures change$" "$OUT/engine.log" ||
    fail "engine: the raise whose read failed was not warned of (see $OUT/engine.log)"
DISPLAY=:$(cat "$OUT/display") "$PYTHON" "$ROOT/hydra/film.py" "$ROOT/hydra/scenes/panned.usda" "$ROOT/hydra/scenes/penumbra.usda" \
    "$ROOT/hydra/scenes/turned.usda" >"$OUT/film.log" 2>&1 ||
    fail "the film's history does not follow a pan, a turn over the sky, objects' motion or a light's change, does not average a soft shadow while playing, or a converged still depends on what came before (see $OUT/film.log)"
clean film "$OUT/film.log"

[ $failed -eq 0 ] && echo "PASS" || exit 1
