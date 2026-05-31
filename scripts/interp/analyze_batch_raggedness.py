#!/usr/bin/env python3
"""Analyze OCR batch raggedness from mokuro timing/summary files.

Usage:
    python scripts/interp/analyze_batch_raggedness.py <summary.json> <timings.jsonl> [bsz]

If bsz is omitted it is read from the summary config.
"""

import json
import math
import sys
from collections import defaultdict


def pctl(data, p):
    """Percentile (linear interpolation)."""
    s = sorted(data)
    n = len(s)
    if n == 0:
        return 0
    k = (n - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def stats_label(data, label=""):
    """Return a dict of common aggregate statistics."""
    if not data:
        return {}
    return {
        "min": min(data),
        "p25": pctl(data, 25),
        "p50": pctl(data, 50),
        "mean": sum(data) / len(data),
        "p75": pctl(data, 75),
        "p90": pctl(data, 90),
        "max": max(data),
        "n": len(data),
    }


def print_stats(label, data, fmt=".2f"):
    """Print a labeled stats block."""
    s = stats_label(data)
    if not s:
        print(f"  {label}: <empty>")
        return s
    bar = f"{label}:"
    print(f"\n  {bar}")
    print(f"    n   : {s['n']}")
    print(f"    min : {format(s['min'], fmt)}")
    print(f"    p25 : {format(s['p25'], fmt)}")
    print(f"    p50 : {format(s['p50'], fmt)}")
    print(f"    mean: {format(s['mean'], fmt)}")
    print(f"    p75 : {format(s['p75'], fmt)}")
    print(f"    p90 : {format(s['p90'], fmt)}")
    print(f"    max : {format(s['max'], fmt)}")
    return s


def histogram(data, lo, hi, bins=10, label=""):
    """Print a simple text histogram."""
    if not data:
        return
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in data:
        idx = min(int((v - lo) / step), bins - 1)
        counts[idx] += 1
    mx = max(counts) if counts else 1
    print(f"\n  {label} (range [{lo:.1f}, {hi:.1f}]):")
    for i, c in enumerate(counts):
        b_lo = lo + i * step
        b_hi = lo + (i + 1) * step
        bar_len = int(c / mx * 40) if mx else 0
        print(f"    [{b_lo:5.1f}, {b_hi:5.1f}): {c:3d} {'█' * bar_len}")


def analyze_summary(summary_path, bsz):
    """Print stats derived from the summary JSON."""
    with open(summary_path) as f:
        summary = json.load(f)

    cfg = summary["config"]
    totals = summary["totals"]
    yield_rate = summary["yield_rate"]
    ragged = summary.get("raggedness", {})

    print("=" * 70)
    print(" CONFIGURATION")
    print("=" * 70)
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    print()
    print("-" * 70)
    print(" TOTALS")
    print("-" * 70)
    for k, v in totals.items():
        print(f"  {k}: {v}")

    # Derived utilization
    mean_batch = yield_rate["batch_size"]["mean"]
    fill = yield_rate["batch_fill_rate"]
    wasted = bsz - mean_batch
    print(f"\n  Effective utilization: {mean_batch:.1f}/{bsz} = {mean_batch/bsz*100:.1f}%")
    print(f"  Avg wasted slots / batch: {wasted:.1f}")

    def print_summary_stats(label, stats_dict, fmt=".2f"):
        """Print pre-aggregated stats from summary JSON (already has min/max/mean/p50/p90)."""
        n = stats_dict.get("count", "?")
        print(f"\n  {label}  (n={n})")
        for key in ("min", "p25", "p50", "mean", "p75", "p90", "max"):
            if key in stats_dict:
                print(f"    {key}: {format(stats_dict[key], fmt)}")

    print()
    print("=" * 70)
    print(" YIELD RATE (from summary)")
    print("=" * 70)
    for metric, stats in yield_rate.items():
        fmt = ".3f" if "rate" in metric else ".1f"
        print_summary_stats(f"  {metric}", stats, fmt=fmt)

    print()
    print("=" * 70)
    print(" RAGGEDNESS (from summary)")
    print("=" * 70)

    for scope_name in ("batches", "reorder_buffers"):
        scope = ragged.get(scope_name, {})
        if not scope:
            continue
        print(f"\n  [{scope_name}]")
        for metric, stats in scope.items():
            print_summary_stats(f"    {metric}", stats, fmt=".1f")


def analyze_raw(jsonl_path, bsz):
    """Reconstruct batches from per-crop timing records and cross-check."""
    with open(jsonl_path) as f:
        crops = [json.loads(line) for line in f if line.strip()]

    print()
    print("=" * 70)
    print(" RAW CROP-LEVEL ANALYSIS (from jsonl)")
    print("=" * 70)
    print(f"\n  Total crop records: {len(crops)}")

    # Group by ocr_batch_size to see what batch sizes actually occurred
    batch_sizes = [c.get("ocr_batch_size", 1) for c in crops]
    unique_bs = sorted(set(batch_sizes))
    print(f"\n  Distinct ocr_batch_size values seen: {unique_bs}")

    # Per-crop token distribution
    tokens = [c["tokens"] for c in crops]
    print_stats("  Tokens per crop", tokens, fmt=".1f")

    histogram(tokens, 0, max(tokens) * 1.1 if tokens else 32, bins=8, label="Token distribution")

    # Group by page to see per-page crop counts
    by_page = defaultdict(list)
    for c in crops:
        by_page[c["page"]].append(c)

    page_crop_counts = [len(v) for v in by_page.values()]
    print_stats("  Crops per page", page_crop_counts, fmt=".0f")

    # Fill-rate distribution (crops-per-page / bsz, capped at 1.0 for pages <= bsz)
    fill_rates = [min(n / bsz, 1.0) for n in page_crop_counts]
    print_stats("  Page fill rate (crops/bsz, capped)", fill_rates, fmt=".3f")

    # Pages that overflow a single batch
    overflow = [(pg, len(crop_list)) for pg, crop_list in by_page.items() if len(crop_list) > bsz]
    if overflow:
        print(f"\n  Pages exceeding bsz={bsz} ({len(overflow)} pages):")
        for pg, n in sorted(overflow, key=lambda x: -x[1])[:10]:
            print(f"    Page {pg}: {n} crops ({n/bsz:.1f}x)")

    # Per-page token raggedness (within-page max/min ratio)
    page_raggedness = []
    for pg, crop_list in by_page.items():
        toks = [c["tokens"] for c in crop_list]
        mn, mx = min(toks), max(toks)
        ratio = mx / mn if mn > 0 else float("inf")
        page_raggedness.append({
            "page": pg,
            "n": len(toks),
            "min_tok": mn,
            "max_tok": mx,
            "ratio": ratio,
        })

    ratios = [p["ratio"] for p in page_raggedness if p["ratio"] != float("inf")]
    print_stats("  Within-page token ratio (max/min)", ratios, fmt=".1f")

    # Top-10 most ragged pages
    worst = sorted(page_raggedness, key=lambda p: -p["ratio"])[:10]
    print(f"\n  Top 10 most ragged pages (by within-page token ratio):")
    for w in worst:
        flag = " ★" if w["ratio"] == float("inf") else ""
        print(
            f"    Page {w['page']:>3}: {w['n']:2d} crops, "
            f"tokens [{w['min_tok']:2d}..{w['max_tok']:2d}], "
            f"ratio={w['ratio']:.1f}x{flag}"
        )

    # Per-page wasted slots (bsz - crops, floored at 0)
    wasted_per_page = [max(0, bsz - n) for n in page_crop_counts]
    print_stats("  Wasted slots per page (bsz - crops, min 0)", wasted_per_page, fmt=".1f")

    # Cross-check: does summary batch count match what we'd expect?
    # With detector_batch_sync, multiple pages feed into shared windows.
    # We can't perfectly reconstruct batches from jsonl alone, but we can
    # estimate: total crops / bsz ≈ expected batch count (last batch partial).
    estimated_batches = math.ceil(len(crops) / bsz)
    print(f"\n  Estimated min batches (total_crops / bsz): {estimated_batches}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    summary_path = sys.argv[1]
    jsonl_path = sys.argv[2]
    bsz = int(sys.argv[3]) if len(sys.argv) > 3 else None

    # Read bsz from summary if not provided
    if bsz is None:
        with open(summary_path) as f:
            bsz = json.load(f)["config"]["ocr_batch_size"]

    analyze_summary(summary_path, bsz)
    analyze_raw(jsonl_path, bsz)


if __name__ == "__main__":
    main()
