#!/usr/bin/env python3
"""Sankey-style failure analysis for OVON navigation experiments.

Reads episode metrics JSON (same format as postprocess_meta.py) and renders
a Sankey diagram showing the flow from all episodes through failure branches.

Usage:
    # With real data
    python failure_analysis_sankey.py --input-path metrics.json --output-path out.html

    # Interactive demo with synthetic data
    python failure_analysis_sankey.py --example --output-path out.html

Output format is determined by the file extension:
    .html  — interactive Plotly (default)
    .png / .svg / .pdf — static image (requires kaleido: pip install kaleido)
"""

import argparse
import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def _get(record: dict, *keys, default=0):
    """Return the first truthy value found among the given keys, else default."""
    for k in keys:
        v = record.get(k, None)
        if v is not None:
            return v
    return default


def compute_failure_stats(records: dict) -> dict:
    """Aggregate episode records into counts for each failure mode.

    Supports both key formats emitted by the codebase:
      - "failure_modes.recognition_failure" / "failure_modes.recognition"
      - bare "recognition_failure" / "recognition"
    """
    total = len(records)
    successes = 0
    mode_counts = {
        "Stop Too Far":        0,  # saw goal, called STOP but was too far away
        "Stop Failure":        0,  # entered success area but never called STOP
        "Recognition Failure": 0,  # saw goal, reached area, but no STOP
        "Misidentification":   0,  # never saw goal but called STOP
        "Exploration":         0,  # never saw goal, never called STOP (timeout)
    }

    for record in records.values():
        if _get(record, "success"):
            successes += 1
            continue

        stop_too_far = _get(
            record,
            "failure_modes.stop_too_far", "stop_too_far",
        )
        stop_failure = _get(
            record,
            "failure_modes.stop_failure", "stop_failure",
            "failure_modes.last_mile_nav",
        )
        recognition = _get(
            record,
            "failure_modes.recognition_failure", "recognition_failure",
            "failure_modes.recognition",
        )
        misidentification = _get(
            record,
            "failure_modes.misidentification", "misidentification",
        )
        exploration = _get(
            record,
            "failure_modes.exploration", "exploration",
        )

        if stop_too_far:
            mode_counts["Stop Too Far"] += 1
        if stop_failure:
            mode_counts["Stop Failure"] += 1
        if recognition:
            mode_counts["Recognition Failure"] += 1
        if misidentification:
            mode_counts["Misidentification"] += 1
        if exploration:
            mode_counts["Exploration"] += 1

    failures = total - successes
    goal_seen_modes = {
        k: mode_counts[k] for k in ("Stop Too Far", "Stop Failure", "Recognition Failure")
    }
    goal_not_seen_modes = {
        k: mode_counts[k] for k in ("Misidentification", "Exploration")
    }
    goal_seen = sum(goal_seen_modes.values())
    goal_not_seen = sum(goal_not_seen_modes.values())

    return {
        "total":             total,
        "successes":         successes,
        "failures":          failures,
        "goal_seen":         goal_seen,
        "goal_not_seen":     goal_not_seen,
        "goal_seen_modes":   goal_seen_modes,
        "goal_not_seen_modes": goal_not_seen_modes,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sankey(stats: dict, output_path: str, title: str = "Navigation Failure Analysis"):
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required: pip install plotly")

    total         = stats["total"]
    successes     = stats["successes"]
    failures      = stats["failures"]
    goal_seen     = stats["goal_seen"]
    goal_not_seen = stats["goal_not_seen"]
    gs_modes      = stats["goal_seen_modes"]
    gns_modes     = stats["goal_not_seen_modes"]

    # ------------------------------------------------------------------
    # Build node list
    # Node indices:
    #   0  All Episodes
    #   1  Success
    #   2  Failure
    #   3  Goal Seen          (saw the target object at least once)
    #   4  Goal Not Seen      (target never visible)
    #   5+ individual failure modes (goal-seen branch)
    #   N+ individual failure modes (goal-not-seen branch)
    # ------------------------------------------------------------------
    base_labels  = ["All Episodes", "Success", "Failure", "Goal Seen", "Goal Not Seen"]
    base_counts  = [total, successes, failures, goal_seen, goal_not_seen]
    base_colors  = ["#4C78A8", "#54A24B", "#E45756", "#F58518", "#9467BD"]

    gs_start  = len(base_labels)
    gs_labels = list(gs_modes.keys())
    gs_counts = list(gs_modes.values())
    gs_colors = ["#FFBF79", "#FFD3B6", "#FF9E9E"]

    gns_start  = gs_start + len(gs_labels)
    gns_labels = list(gns_modes.keys())
    gns_counts = list(gns_modes.values())
    gns_colors = ["#C5B0D5", "#DBDB8D"]

    all_labels = base_labels + gs_labels + gns_labels
    all_counts = base_counts + gs_counts + gns_counts
    all_colors = base_colors + gs_colors[:len(gs_labels)] + gns_colors[:len(gns_labels)]

    pct_labels = []
    for label, count in zip(all_labels, all_counts):
        pct = f"{count / total * 100:.1f}%" if total > 0 else ""
        pct_labels.append(f"{label}<br><b>{count}</b> ({pct})")

    # ------------------------------------------------------------------
    # Build links
    # ------------------------------------------------------------------
    src, tgt, val, link_colors = [], [], [], []

    def add_link(s, t, v, color):
        if v > 0:
            src.append(s); tgt.append(t); val.append(v); link_colors.append(color)

    # All Episodes → Success / Failure
    add_link(0, 1, successes,   "rgba(84,162,75,0.4)")
    add_link(0, 2, failures,    "rgba(228,87,86,0.4)")

    # Failure → Goal Seen / Goal Not Seen
    add_link(2, 3, goal_seen,     "rgba(245,133,24,0.4)")
    add_link(2, 4, goal_not_seen, "rgba(148,103,189,0.4)")

    # Goal Seen → individual modes
    gs_link_colors = ["rgba(255,191,121,0.5)", "rgba(255,211,182,0.5)", "rgba(255,158,158,0.5)"]
    for i, (mode, count) in enumerate(gs_modes.items()):
        add_link(3, gs_start + i, count, gs_link_colors[i % len(gs_link_colors)])

    # Goal Not Seen → individual modes
    gns_link_colors = ["rgba(197,176,213,0.5)", "rgba(219,219,141,0.5)"]
    for i, (mode, count) in enumerate(gns_modes.items()):
        add_link(4, gns_start + i, count, gns_link_colors[i % len(gns_link_colors)])

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=25,
            thickness=22,
            line=dict(color="black", width=0.5),
            label=pct_labels,
            color=all_colors,
        ),
        link=dict(
            source=src,
            target=tgt,
            value=val,
            color=link_colors,
        ),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        font=dict(size=13, family="Arial"),
        height=520,
        width=1000,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
    )

    output_path = Path(output_path)
    ext = output_path.suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".svg", ".pdf"):
        try:
            fig.write_image(str(output_path))
        except Exception:
            # Kaleido requires Chrome; fall back to matplotlib static render.
            html_path = output_path.with_suffix(".html")
            fig.write_html(str(html_path))
            print(
                f"Note: static image export requires Chrome "
                f"(`plotly_get_chrome` to install it). "
                f"Saved interactive HTML instead → {html_path}"
            )
            output_path = html_path
    else:
        output_path = output_path.with_suffix(".html")
        fig.write_html(str(output_path))

    print(f"Saved → {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Example / demo data
# ---------------------------------------------------------------------------

def generate_example_data(n: int = 500, seed: int = 42) -> dict:
    """Synthetic records that mimic the JSON produced by the eval pipeline."""
    random.seed(seed)
    records = {}
    thresholds = [
        (0.44, {"success": 1}),
        (0.59, {"success": 0, "failure_modes.stop_too_far": 1}),
        (0.68, {"success": 0, "failure_modes.recognition_failure": 1}),
        (0.76, {"success": 0, "failure_modes.stop_failure": 1}),
        (0.90, {"success": 0, "failure_modes.exploration": 1}),
        (1.00, {"success": 0, "failure_modes.misidentification": 1}),
    ]
    for i in range(n):
        r = random.random()
        for threshold, record in thresholds:
            if r < threshold:
                records[str(i)] = record
                break
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sankey failure analysis for OVON navigation experiments."
    )
    parser.add_argument(
        "--input-path", "-i",
        type=str,
        help="Path to episode metrics JSON file.",
    )
    parser.add_argument(
        "--output-path", "-o",
        type=str,
        default="failure_analysis.html",
        help="Output file (.html for interactive, .png/.svg/.pdf for static).",
    )
    parser.add_argument(
        "--title", "-t",
        type=str,
        default="Navigation Failure Analysis",
        help="Plot title.",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run with synthetic example data (no --input-path required).",
    )
    parser.add_argument(
        "--n-example",
        type=int,
        default=500,
        help="Number of synthetic episodes when using --example.",
    )
    args = parser.parse_args()

    if args.example or args.input_path is None:
        print(f"Generating {args.n_example} synthetic episodes for demonstration…")
        records = generate_example_data(n=args.n_example)
    else:
        records = load_json(args.input_path)

    stats = compute_failure_stats(records)
    print("\nAggregate stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    plot_sankey(stats, args.output_path, title=args.title)
