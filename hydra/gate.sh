#!/usr/bin/env bash
# The gate run, on your own display: ALab (corpus/fetch.sh) in usdview at 1080p, frames 1004-1057 played twice
# (hydra/gate.py); the second loop's period p95 and misses are the gate's numbers. The
# validation layer stays off: it would be timed too.
#
# The host is the flake's usdview, or Houdini's when HFS names one.
#
#   hydra/gate.sh <out dir>
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
STAGE=$ROOT/corpus/ALab-2.3.0/ALab/entry.usda
OUT=${1:?usage: gate.sh <out dir>}
[ -e "$STAGE" ] || { echo "ALab is not in corpus/: run corpus/fetch.sh"; exit 1; }
CAMERA=/root/camera01/GEO/renderCam_hrc/renderCam_buffer/renderCam_srt/renderCam

make -C "$ROOT/hydra" --no-print-directory ${HFS:+HFS="$HFS"}
mkdir -p "$OUT"
if [ -n "${HFS:-}" ]; then
    # houdini_setup_bash reads unset variables.
    set +u
    cd "$HFS" && source ./houdini_setup_bash >/dev/null && cd "$ROOT"
    set -u
fi
export PXR_PLUGINPATH_NAME=$ROOT/target/hydra
# The flake's shell names the validation layer, which Pika turns on when it finds it.
unset VK_ADD_LAYER_PATH

# The driver lowers clocks while the GPU is mostly idle, so runs compare
# only at the same clocks.
nvidia-smi --query-gpu=timestamp,clocks.gr,clocks.mem,pstate,utilization.gpu,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv,noheader -lms 200 >"$OUT/clocks.csv" &
smi=$!
trap 'kill $smi' EXIT
# Hosts start with the usual soft limit of 1024 open files; a shell raised past it
# would hide a leak.
ulimit -Sn 1024
PIKA_STATS=$OUT/gate.stats testusdview --renderer Pika --camera "$CAMERA" \
    --sessionLayer "$ROOT/hydra/scenes/alab_nofur.usda" --testScript "$ROOT/hydra/gate.py" \
    "$STAGE" >"$OUT/gate.log" 2>&1 || { echo "usdview exited $? (see $OUT/gate.log)"; exit 1; }
if grep "^pika: error" "$OUT/gate.log"; then
    echo "Pika reported errors (see $OUT/gate.log)"
    exit 1
fi
# A material drawn pink is cheaper than the one the scene asked for.
if grep -q "drawn pink" "$OUT/gate.log"; then
    echo "$(grep -c "drawn pink" "$OUT/gate.log") materials were drawn pink (see $OUT/gate.log)"
    exit 1
fi
python3 -I "$ROOT/hydra/stats.py" "$OUT/gate.stats" 54
awk -F', ' '{ print $2 + 0 }' "$OUT/clocks.csv" | sort -n |
    awk '{ c[NR] = $1 } END { print "graphics clock (MHz): min " c[1] ", median " c[int(NR / 2) + 1] ", max " c[NR] }'
