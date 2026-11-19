#!/usr/bin/env bash
# Fetches ALab 2.3.0 (https://dpel.aswf.io/alab/), 108 GB of archives, and assembles it with ALab's
# own installer into corpus/ALab-2.3.0, where hydra/gate.sh plays it.
# Downloading accepts the ASWF Digital Assets License v1.1.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ALAB=$HERE/ALab-2.3.0
COMMIT=d191bf4875b150198c6c69040bedf75509f93729
CAMERA=/root/camera01/GEO/renderCam_hrc/renderCam_buffer/renderCam_srt/renderCam
[ -e "$ALAB/assembled" ] && { echo "$ALAB is already assembled"; exit 0; }
mkdir -p "$HERE/archives"
cd "$HERE/archives"

pack() {
    [ -e "$1.ok" ] && return
    curl -L --fail -C - -o "$1" "https://$3/usd-alab/$1"
    echo "$2  $1" | sha256sum -c -
    touch "$1.ok"
}

if [ ! -e structure.zip.ok ]; then
    curl -L --fail -o structure.zip \
        https://github.com/DigitalProductionExampleLibrary/ALab/archive/refs/tags/v2.3.0.zip
    [ "$(unzip -z structure.zip | tail -n 1)" = "$COMMIT" ] ||
        { echo "structure.zip is not ALab's commit $COMMIT"; exit 1; }
    touch structure.zip.ok
fi
pack alab-techvars.v2.2.0.zip d142891ed4ad2365f8dd6b583e9dac88982131d7709dd8e6f6bc0f1516007ec6 dpel-assets.aswf.io
pack alab-procedurals.v2.2.0.zip 547dd58fa8d3cb712466137380d5b1a2686a572efb8dabe99d74012e14fe930c dpel-assets.aswf.io
pack alab-cameras.v2.2.0.zip 970cc9f69567fb880f48af737e1c06fdd5b2ef14feb40f902173629bc5d4c9f4 dpel-assets.aswf.io
pack alab-textures.v2.2.0.zip 98eed08292f243f73571b0f8d7bba35375d54fb07fa4cef70333d94ec8dea20d aswf-dpel-assets.s3.amazonaws.com

unzip -q -o structure.zip -d "$HERE"
python3 "$ALAB/install_optional_packages.py" --output "$ALAB" --techvar alab-techvars.v2.2.0.zip \
    --baked_procedurals alab-procedurals.v2.2.0.zip --cameras alab-cameras.v2.2.0.zip \
    --texture_pack alab-textures.v2.2.0.zip

python3 -c '
import sys
from pxr import Usd
stage = Usd.Stage.Open(sys.argv[1])
frames = stage.GetStartTimeCode(), stage.GetEndTimeCode()
if not stage.GetPrimAtPath(sys.argv[2]) or frames != (1004, 1057):
    sys.exit(f"{sys.argv[1]} is not the scene the gate plays: frames {frames}, camera {sys.argv[2]}")
' "$ALAB/ALab/entry.usda" "$CAMERA"
touch "$ALAB/assembled"
echo "ALab is in $ALAB. Netflix Animation Studios ALab Copyright 2025 Netflix, Inc. All rights reserved."
