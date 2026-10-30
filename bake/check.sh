#!/usr/bin/env bash
# The bake tool's check, run where Houdini 22 and slangc are (not in CI).
#
# Builds the tool and generates bake/fixtures/network.mtlx, the document
# src/material.rs writes for hydra/fixtures/network.usda. The manifest must
# match bake/fixtures/network.manifest, and the pixel stage must compile
# with network.surface.slang, the vertex stage src/surface.rs writes for
# it, as Pika compiles them, reflecting the layout in
# network.reflection.json that src/surface.rs's tests pack against. A
# document naming a nodedef the libraries lack must fail, naming it.
#
#   bake/check.sh
set -euo pipefail

HFS=${HFS:-/opt/hfs22.0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/target/bake/check
BAKE=$ROOT/target/bake/bake

make -C "$ROOT/bake" --no-print-directory HFS="$HFS"
rm -rf "$OUT"
mkdir -p "$OUT/network" "$OUT/missing"

failed=0
fail() {
    echo "FAIL: $*"
    failed=1
}

"$BAKE" generate "$ROOT/bake/fixtures/network.mtlx" "$OUT/network" ||
    fail "network: generate failed"
diff "$ROOT/bake/fixtures/network.manifest" "$OUT/network/manifest" ||
    fail "network: the manifest differs from the reference"
# The generator declares textures inside its uniform block, and slangc warns
# that it moves them to bindings of their own, which is where Pika binds them.
cp "$ROOT/bake/fixtures/network.surface.slang" "$OUT/network/surface.slang"
slangc "$OUT/network/surface.slang" -I "$ROOT/shaders" -target spirv -fvk-use-entrypoint-name \
    -warnings-as-errors all -warnings-disable 31106,31107 -o "$OUT/network/surface.spv" \
    -reflection-json "$OUT/network/reflection.json" ||
    fail "network: slangc rejected the surface"
diff "$ROOT/bake/fixtures/network.reflection.json" "$OUT/network/reflection.json" ||
    fail "network: the reflection differs from the reference"

sed 's/ND_geompropvalue_vector2/ND_missing/' "$ROOT/bake/fixtures/network.mtlx" > "$OUT/missing.mtlx"
if "$BAKE" generate "$OUT/missing.mtlx" "$OUT/missing" 2> "$OUT/missing.err"; then
    fail "missing: a document with an unknown nodedef generated"
fi
grep -q "ND_missing" "$OUT/missing.err" || fail "missing: the error does not name the nodedef"

[ $failed = 0 ] && echo "PASS: bake"
exit $failed
