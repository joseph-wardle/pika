"""Summarises a PIKA_STATS file: every time as p50/p95/max over the warm
syncs and frames, the frames over the gate's period, each budget line's p95
against its budget, and the last frame's memory.

    python3 hydra/stats.py <file.stats> [<warm>]

Pika appends one line per Hydra sync and per rendered frame: the kind and
index, then `key value` pairs (src/stats.rs). A key ending in `_ms` is
milliseconds; any other value is a count. The warm window is the last <warm>
syncs and frames; without it, all but the first, or the only one. A loop
played twice over 54 frames summarises with 54.
"""

import sys

# Over 1080p at 24 fps: the gate's frame period.
BUDGET_MS = 41.67
# Mesh drawing's budget lines, each the sum of the frame keys named: the
# warm p95 on ALab in usdview at 1080p. They are measured lines, not limits;
# a line over its budget moves work up the order (design.md).
LINES = (
    ("draw", ("gpu_draw_ms",), 4.0),
    ("update", ("gpu_update_ms",), 1.0),
    ("delivery", ("gpu_readback_ms", "place_ms"), 5.0),
    ("cpu", ("record_ms", "wait_ms"), 2.0),
    ("period", ("period_ms",), BUDGET_MS),
)


def read(path):
    """The file's host and device, and its sync and frame records, each a
    dict of key to float. The frames must all have the first frame's keys
    (the first's missing period aside), and the syncs the first sync's,
    counted prim types aside, so a line cut short fails here."""
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
    """A record's keys, without those that may differ between records."""
    return {k for k in row if k != "period_ms" and not k.startswith("counted_")}


def percentiles(values):
    """p50, p95 and max, nearest rank."""
    values = sorted(values)
    rank = lambda p: values[min(len(values) - 1, int(p * len(values)))]
    return rank(0.5), rank(0.95), values[-1]


def misses(periods):
    """How many periods exceed the budget, and the longest run of them."""
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
          f"{last['drawn_instances']:.0f} instances, {last['drawn_triangles'] / 1e6:.2f} M triangles")
    print(f"  memory: geometry {size(last['geometry_bytes'])}, views {size(last['view_bytes'])}, "
          f"targets {size(last['target_bytes'])}, host {size(last['host_bytes'])}; "
          f"vram {size(last['vram_used_bytes'])} used of {size(last['vram_budget_bytes'])}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(*sys.argv[1:])
