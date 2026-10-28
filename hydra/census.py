"""The census: syncs a shot uncached through Pika in husk, frame by frame,
and says which renderer work its costs put first.

    hython hydra/census.py <stage.usda> <camera> <first frame> <last frame> <out dir>

Run it from a shell with houdini_setup_bash sourced and PXR_PLUGINPATH_NAME
pointing at the delegate. It syncs the shot three times: twice without its
basis curves (the gate's scene, as fur is out of the gate), first with the
stage's value clip files evicted from this machine's page cache and then with
them cached by that run, and once whole, to measure what fur costs. Each run's
per-sync lines are in <out dir>/<run>.stats.

The thresholds were fixed before the first run, in design.md's census entry.
The first sync loads the scene, so the warm figures leave it out. Clip files
are where each frame's animation is read; read cold from NFS they made ALab's
sync two to three times slower than cached, so storage gets its own verdict.
"""

import os
import subprocess
import sys

from pxr import Sdf, Usd, UsdGeom

# Importing the oracle would leave a __pycache__ in the checkout.
sys.dont_write_bytecode = True
from oracle import over  # noqa: E402

# A sync sub-budget, from the feasibility study's frame budget.
SYNC_P95_MS = 10.42
# Of the bytes Pika stores, the share Hydra hands over unchanged.
REDUNDANT_SHARE = 0.25
# A draw call per mesh instance per pass, and triangles before subdivision.
MESH_INSTANCES = 10_000
TRIANGLES = 20_000_000


def write_gate_layer(stage, path):
    """A layer over the stage deactivating its basis curves. Returns them."""
    layer = over(stage)
    curves = [p.GetPath() for p in stage.Traverse() if p.IsA(UsdGeom.BasisCurves)]
    for curve in curves:
        Sdf.CreatePrimInLayer(layer, curve).active = False
    layer.Export(path)
    return curves


def clip_files(stage):
    """The files of the stage's value clips."""
    files = set()
    for prim in stage.TraverseAll():
        for clips in Usd.ClipsAPI(prim).GetClips().values():
            files.update(p.resolvedPath for p in clips.get("assetPaths", []) if p.resolvedPath)
    return sorted(files)


def evict(paths):
    """Drops the files' pages from this machine's page cache, so the next read
    goes to storage. A file server's own cache is out of reach."""
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def sync(name, stage, camera, first, last, out):
    """Runs husk over the frames; returns the stats lines and husk's peak
    resident memory in GiB."""
    stats = os.path.join(out, f"{name}.stats")
    log = os.path.join(out, f"{name}.log")
    os.makedirs(os.path.join(out, name), exist_ok=True)
    command = ["husk", "-V", "1", "-R", "Pika", "--headlight", "none",
               "-f", str(first), "-n", str(last - first + 1), "--res", "64", "64",
               "-c", camera, "-o", os.path.join(out, name, "frame.$F4.exr"), stage]
    with open(log, "w") as f:
        child = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, PIKA_STATS=stats))
        _, status, usage = os.wait4(child.pid, 0)
    if os.waitstatus_to_exitcode(status) != 0:
        sys.exit(f"census: husk exited {os.waitstatus_to_exitcode(status)} (see {log})")
    if any(line.startswith("pika: error") for line in open(log)):
        sys.exit(f"census: Pika reported errors (see {log})")
    rows = []
    for line in open(stats):
        words = line.split()
        rows.append({k: float(v) for k, v in zip(words[::2], words[1::2])})
    if len(rows) < last - first + 1:
        sys.exit(f"census: {len(rows)} syncs for {last - first + 1} frames (see {log})")
    return rows, usage.ru_maxrss / 2**20


def percentiles(values):
    """p50, p95 and max, nearest rank."""
    values = sorted(values)
    rank = lambda p: values[min(len(values) - 1, int(p * len(values)))]
    return rank(0.5), rank(0.95), values[-1]


def ms(values):
    return "p50 %.2f, p95 %.2f, max %.2f ms" % percentiles(values)


def mb(values):
    return "p50 %.1f, max %.1f MB" % percentiles(values)[::2]


def report(name, rows, rss):
    """Prints a run's figures; returns the warm sync p95 and redundant share."""
    cold, warm = rows[0], rows[1:]
    stored = lambda r: r["mesh_bytes"] + r["instancer_bytes"]
    redundant = sum(r["mesh_redundant_bytes"] + r["instancer_redundant_bytes"] for r in warm)
    share = redundant / max(1, sum(stored(r) for r in warm))
    last = rows[-1]
    print(f"\n{name}: {len(rows)} syncs, husk peak {rss:.1f} GiB resident")
    print(f"  first sync {cold['time_ms']:.0f} ms, {stored(cold) / 1e6:.1f} MB stored, "
          f"{cold['curve_bytes'] / 1e6:.1f} MB of curves")
    print(f"  warm sync    {ms(r['time_ms'] for r in warm)}")
    print(f"  warm period  {ms(r['period_ms'] for r in warm)} (husk, sync to sync)")
    print(f"  mesh bytes      {mb([r['mesh_bytes'] / 1e6 for r in warm])} per sync")
    print(f"  instancer bytes {mb([r['instancer_bytes'] / 1e6 for r in warm])} per sync")
    print(f"  curve bytes     {mb([r['curve_bytes'] / 1e6 for r in warm])} per sync")
    print(f"  redundant {100 * share:.1f}% of the bytes stored")
    print(f"  {last['meshes']:.0f} meshes drawn at {last['mesh_instances']:.0f} instances, "
          f"{last['triangles'] / 1e6:.2f} M triangles "
          f"({last['prototype_triangles'] / 1e6:.2f} M once each, "
          f"{last['geometry_bytes'] / 2**20:.0f} MiB of geometry)")
    print(f"  {last['materials']:.0f} materials bound, {last['instancers']:.0f} instancers, "
          f"{last['curves']:.0f} basis curves")
    counted = {k[len("counted_"):]: int(v) for k, v in last.items() if k.startswith("counted_")}
    print("  counted: " + ", ".join(f"{v} {k}" for k, v in sorted(counted.items())))
    return percentiles([r["time_ms"] for r in warm])[1], share, last


def verdict(exceeded, yes, no):
    print(f"  {'OVER ' if exceeded else 'under'}  {yes if exceeded else no}")


def main(stage_path, camera, first, last, out):
    first, last = int(first), int(last)
    os.makedirs(out, exist_ok=True)
    stage = Usd.Stage.Open(stage_path)
    gate = os.path.join(out, "gate.usda")
    curves = write_gate_layer(stage, gate)
    clips = clip_files(stage)
    print(f"census: {stage_path}, frames {first}-{last}, camera {camera}; "
          f"the gate runs deactivate {len(curves)} basis curves; {len(clips)} clip files")

    evict(clips)
    cold, _, _ = report("gate, clip files cold", *sync("gate_cold", gate, camera, first, last, out))
    p95, share, census = report("gate, clip files cached",
                                *sync("gate", gate, camera, first, last, out))
    report("with fur", *sync("fur", stage_path, camera, first, last, out))

    print("\nverdicts (gate runs):")
    verdict(p95 > SYNC_P95_MS,
            f"warm sync p95 {p95:.2f} ms > {SYNC_P95_MS} ms with clip files cached: "
            "the playback cache comes first",
            f"warm sync p95 {p95:.2f} ms <= {SYNC_P95_MS} ms with clip files cached: no cache yet")
    verdict(cold > SYNC_P95_MS,
            f"warm sync p95 {cold:.2f} ms > {SYNC_P95_MS} ms with clip files cold: "
            "a cache cannot help the first pass; prefetch clip files or play from local disk",
            f"warm sync p95 {cold:.2f} ms <= {SYNC_P95_MS} ms with clip files cold: "
            "storage keeps up")
    verdict(share > REDUNDANT_SHARE,
            f"{100 * share:.1f}% redundant > {100 * REDUNDANT_SHARE:.0f}%: fix the asset first",
            f"{100 * share:.1f}% redundant <= {100 * REDUNDANT_SHARE:.0f}%: no asset fix")
    verdict(census["mesh_instances"] > MESH_INSTANCES,
            f"{census['mesh_instances']:.0f} mesh instances > {MESH_INSTANCES}: "
            "instanced drawing comes first",
            f"{census['mesh_instances']:.0f} mesh instances <= {MESH_INSTANCES}: a draw per instance")
    verdict(census["triangles"] > TRIANGLES,
            f"{census['triangles'] / 1e6:.1f} M triangles > {TRIANGLES / 1e6:.0f} M: "
            "a geometry budget comes first",
            f"{census['triangles'] / 1e6:.1f} M triangles <= {TRIANGLES / 1e6:.0f} M: "
            "no geometry budget yet")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    main(*sys.argv[1:])
