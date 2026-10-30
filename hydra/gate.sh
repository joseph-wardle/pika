#!/usr/bin/env bash
# The gate run, on your own display: ALab in usdview, its trailer camera
# fitted into 1080p, the render purpose, fur off (hydra/scenes/alab_nofur.usda),
# frames 1004-1057 played twice as the Play button does (hydra/gate.py).
# Pika writes PIKA_STATS, nvidia-smi logs the clocks beside it, and the
# second loop is summarised with hydra/stats.py: its period p95 and misses
# are the gate's numbers, and each budget line's p95 is shown against its
# budget. The validation layer stays off: it would be timed too.
#
#   hydra/gate.sh /path/to/ALab/entry.usda <out dir>
set -euo pipefail

HFS=${HFS:-/opt/hfs21.0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
STAGE=${1:?usage: gate.sh <ALab entry.usda> <out dir>}
OUT=${2:?usage: gate.sh <ALab entry.usda> <out dir>}
CAMERA=/root/camera01/GEO/renderCam_hrc/renderCam_buffer/renderCam_srt/renderCam

make -C "$ROOT/hydra" --no-print-directory
mkdir -p "$OUT"
# houdini_setup_bash reads unset variables.
set +u
cd "$HFS" && source ./houdini_setup_bash >/dev/null && cd "$ROOT"
set -u
export PXR_PLUGINPATH_NAME=$ROOT/target/hydra

# The driver lowers clocks while the GPU is mostly idle, so runs compare
# only at the same clocks.
nvidia-smi --query-gpu=timestamp,clocks.gr,clocks.mem,pstate,utilization.gpu,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv,noheader -lms 200 >"$OUT/clocks.csv" &
smi=$!
trap 'kill $smi' EXIT
PIKA_STATS=$OUT/gate.stats testusdview --renderer Pika --camera "$CAMERA" \
    --sessionLayer "$ROOT/hydra/scenes/alab_nofur.usda" --testScript "$ROOT/hydra/gate.py" \
    "$STAGE" >"$OUT/gate.log" 2>&1 || { echo "usdview exited $? (see $OUT/gate.log)"; exit 1; }
if grep "^pika: error" "$OUT/gate.log"; then
    echo "Pika reported errors (see $OUT/gate.log)"
    exit 1
fi
python3 -I "$ROOT/hydra/stats.py" "$OUT/gate.stats" 54
awk -F', ' '{ print $2 + 0 }' "$OUT/clocks.csv" | sort -n |
    awk '{ c[NR] = $1 } END { print "graphics clock (MHz): min " c[1] ", median " c[int(NR / 2) + 1] ", max " c[NR] }'
