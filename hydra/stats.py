"""Summarises a PIKA_STATS file: each time as p50/p95/max over the warm syncs and frames, the
frames over the gate's period, each budget line's p95 against its budget, and the last frame's
memory.

    python3 hydra/stats.py <file.stats> [<warm>]

Each line is a sync or frame, its index, then `key value` pairs (src/stats.rs). `_ms` keys are
milliseconds, the rest counts. Given <warm>, the last <warm> syncs and frames are claimed warm,
and one still building a material or texture fails. Otherwise the window is all but the first.
A loop played twice over 54 frames summarises with 54.
"""

import sys

# 24 fps: the gate's frame period.
BUDGET_MS = 41.67
# Sums of the frame keys named. Measured warm p95 on ALab in usdview at 1080p, not limits.
LINES = (
    ("draw", ("gpu_draw_ms",), 4.0),
    ("shade", ("gpu_shade_ms",), 12.0),
    # Listing, drawing and copying pages. Their lookups are in shade.
    ("shadow", ("gpu_shadow_list_ms", "gpu_shadow_pages_ms", "gpu_shadow_copy_ms",
                "gpu_shadow_live_ms"), 3.0),
    # The copies out of the attachments, history's resolve or the still sum, and the copy back.
    ("film", ("gpu_film_ms",), 1.0),
    ("update", ("gpu_upload_ms", "gpu_subdiv_ms", "gpu_smooth_ms", "gpu_tangent_ms",
                "gpu_texture_ms"), 1.0),
    ("delivery", ("gpu_readback_ms", "place_ms"), 5.0),
    ("cpu", ("record_ms", "wait_ms"), 2.0),
    ("period", ("period_ms",), BUDGET_MS),
)
BUILDING = ("pending_materials", "pending_textures", "compiled_surfaces", "baked_textures")


def read(path):
    """Records must have the first's keys, the first frame's missing period and counted prim types
    aside, so a line cut short fails here."""
    host = device = ""
    records = {"sync": [], "frame": []}
    for line in open(path):
        words = line.split()
        if not words:
            continue
        kind = words[0]
        if kind in ("host", "device"):
            if kind == "host":
                host = " ".join(words[1:])
            else:
                device = " ".join(words[1:])
            continue
        if kind not in records:
            raise ValueError(f"{path}: unknown line kind {kind!r}")
        if len(words) % 2:
            raise ValueError(f"{path}: {kind} {words[1]} has a key without a value")
        row = {k: float(v) for k, v in zip(words[2::2], words[3::2])}
        rows = records[kind]
        if rows and keys(rows[0]) != keys(row):
            raise ValueError(f"{path}: {kind} {words[1]} has different keys from {kind} 1")
        rows.append(row)
    return host, device, records["sync"], records["frame"]


def keys(row):
    return {k for k in row if k != "period_ms" and not k.startswith("counted_")}


def percentiles(values):
    """Nearest rank."""
    values = sorted(values)
    rank = lambda p: values[min(len(values) - 1, int(p * len(values)))]
    return rank(0.5), rank(0.95), values[-1]


def misses(periods):
    over = run = longest = 0
    for period in periods:
        run = run + 1 if period > BUDGET_MS else 0
        over += period > BUDGET_MS
        longest = max(longest, run)
    return over, longest


def size(b):
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.1f} MB"
    return f"{b:.0f} B"


def main(path, warm=None):
    host, device, syncs, frames = read(path)
    window = (lambda rows: rows[-int(warm):]) if warm else (lambda rows: rows[1:] or rows)
    s, f = window(syncs), window(frames)
    busy = [k for k in BUILDING if warm and any(r[k] for r in f)]
    if busy:
        sys.exit(f"{path}: the warm window is not warm: {', '.join(busy)} in its frames")
    print(f"{path}: {len(syncs)} syncs, {len(frames)} frames on {device} ({host})")
    print(f"warm window: {len(s)} syncs, {len(f)} frames; ms as p50/p95/max")
    for rows, prefix in ((s, "sync_"), (f, "")):
        for key in rows[-1] if rows else ():
            if key.endswith("_ms"):
                values = [r[key] for r in rows if key in r]
                print("  %-20s %8.2f %8.2f %8.2f" % ((prefix + key[:-3],) + percentiles(values)))
    if not f:
        return
    over, run = misses(r["period_ms"] for r in f if "period_ms" in r)
    last = f[-1]
    print(f"  frames over {BUDGET_MS} ms: {over}, longest run {run}")
    for name, keys_of_line, budget in LINES:
        values = [sum(r[k] for k in keys_of_line) for r in f if all(k in r for k in keys_of_line)]
        if values:
            p95 = percentiles(values)[1]
            print(f"  {name:<10s} p95 {p95:8.2f} of {budget:g} ms: {'met' if p95 <= budget else 'OVER'}")
    print(f"  last frame: {last['width']:.0f}x{last['height']:.0f}, "
          f"{last['drawn_instances']:.0f} instances, {last['drawn_triangles'] / 1e6:.2f} M triangles, "
          f"{last['drawn_lights']:.0f} lights")
    print(f"  memory: geometry {size(last['geometry_bytes'])}, views {size(last['view_bytes'])}, "
          f"targets {size(last['target_bytes'])}, host {size(last['host_bytes'])}; "
          f"vram {size(last['vram_used_bytes'])} used of {size(last['vram_budget_bytes'])}")
    print(f"  materials: {size(last['material_bytes'])}, {last['lowered_textures']:.0f} textures lowered to fit")
    print(f"  shadows: {size(last['shadow_bytes'])}, {last['shadow_pages_held']:.0f} pages held of "
          f"{last['shadow_pages_wanted']:.0f} wanted, {last['shadow_draws']:.0f} caster draws")
    print(f"  textures: {last['blurred_textures']:.0f} drawn coarser than the view needs, "
          f"{last['unseen_textures']:.0f} unseen, {last['unestimated_textures']:.0f} not estimated")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(*sys.argv[1:])
