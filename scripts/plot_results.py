"""Generate comparison plots from experiment results.

Usage:
    python scripts/plot_results.py                    # default: all results
    python scripts/plot_results.py --output paper/figures/
    python scripts/plot_results.py --groups I M        # only IFR and Moslem groups
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.aggregate_results import collect_results


# ── Styling ──────────────────────────────────────────────────────────────────

GROUP_COLORS = {
    "B": "#4C72B0",   # baselines: blue
    "M": "#DD8452",   # Moslem heuristic: orange
    "I": "#55A868",   # IFR-guided: green
    "L": "#C44E52",   # LRP-guided: red
}

GROUP_LABELS = {
    "B": "Baseline",
    "M": "Heuristic (Moslem)",
    "I": "IFR-guided",
    "L": "LRP-guided",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def get_group(exp_id: str) -> str:
    """Extract group prefix (B, M, I, L) from experiment ID."""
    for ch in exp_id:
        if ch.isalpha():
            return ch.upper()
    return "?"


def color_for(exp_id: str) -> str:
    return GROUP_COLORS.get(get_group(exp_id), "#999999")


# ── Plot Functions ───────────────────────────────────────────────────────────

def plot_metric_bars(df: pd.DataFrame, metric: str, ylabel: str, title: str,
                     ax: plt.Axes, baseline_val: float = None):
    """Bar chart for a single metric across experiments."""
    ids = df["experiment_id"].tolist()
    vals = df[metric].tolist()
    colors = [color_for(eid) for eid in ids]

    bars = ax.bar(range(len(ids)), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add baseline reference line
    if baseline_val is not None:
        ax.axhline(baseline_val, color="#4C72B0", linestyle="--", linewidth=1,
                    alpha=0.7, label=f"Baseline ({baseline_val:.2f})")
        ax.legend(fontsize=8, loc="best")

    # Value labels on bars
    for bar, val in zip(bars, vals):
        if pd.notna(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)


def plot_quality_metrics(df: pd.DataFrame, output_dir: Path):
    """Combined bar chart: COMET, chrF++, BLEU side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    baseline = df[df["experiment_id"] == "B0"]
    b_comet = baseline["comet"].iloc[0] if not baseline.empty else None
    b_chrf = baseline["chrf"].iloc[0] if not baseline.empty else None
    b_bleu = baseline["bleu"].iloc[0] if not baseline.empty else None

    plot_metric_bars(df, "comet", "COMET", "COMET Score", axes[0], b_comet)
    plot_metric_bars(df, "chrf", "chrF++", "chrF++ Score", axes[1], b_chrf)
    plot_metric_bars(df, "bleu", "BLEU", "BLEU Score", axes[2], b_bleu)

    # Add group legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=GROUP_LABELS.get(g, g))
                       for g, c in GROUP_COLORS.items()
                       if g in df["experiment_id"].apply(get_group).unique()]
    fig.legend(handles=legend_elements, loc="upper center",
               ncol=len(legend_elements), fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Translation Quality Metrics", fontsize=14, y=1.06)
    plt.tight_layout()
    fig.savefig(output_dir / "quality_metrics.png")
    plt.close(fig)
    print(f"  Saved quality_metrics.png")


def plot_efficiency(df: pd.DataFrame, output_dir: Path):
    """Scatter plot: quality (COMET) vs efficiency (model size, speed)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for _, row in df.iterrows():
        eid = row["experiment_id"]
        g = get_group(eid)
        c = color_for(eid)

        if pd.notna(row.get("size_mb")) and pd.notna(row.get("comet")):
            axes[0].scatter(row["size_mb"], row["comet"], c=c, s=80,
                            edgecolors="white", linewidth=0.5, zorder=3)
            axes[0].annotate(eid, (row["size_mb"], row["comet"]),
                             fontsize=7, ha="left", va="bottom",
                             xytext=(4, 4), textcoords="offset points")

        if pd.notna(row.get("tokens_per_sec")) and pd.notna(row.get("comet")):
            axes[1].scatter(row["tokens_per_sec"], row["comet"], c=c, s=80,
                            edgecolors="white", linewidth=0.5, zorder=3)
            axes[1].annotate(eid, (row["tokens_per_sec"], row["comet"]),
                             fontsize=7, ha="left", va="bottom",
                             xytext=(4, 4), textcoords="offset points")

    axes[0].set_xlabel("Model Size (MB)")
    axes[0].set_ylabel("COMET")
    axes[0].set_title("Quality vs Model Size")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Tokens / Second")
    axes[1].set_ylabel("COMET")
    axes[1].set_title("Quality vs Inference Speed")
    axes[1].grid(True, alpha=0.3)

    # Group legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=GROUP_LABELS.get(g, g))
                       for g, c in GROUP_COLORS.items()
                       if g in df["experiment_id"].apply(get_group).unique()]
    fig.legend(handles=legend_elements, loc="upper center",
               ncol=len(legend_elements), fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Quality–Efficiency Trade-off", fontsize=14, y=1.06)
    plt.tight_layout()
    fig.savefig(output_dir / "efficiency_tradeoff.png")
    plt.close(fig)
    print(f"  Saved efficiency_tradeoff.png")


def plot_layers_vs_quality(df: pd.DataFrame, output_dir: Path):
    """Line/scatter: number of remaining layers vs quality metrics."""
    df_with_layers = df[df["num_layers"].notna() & (df["num_layers"] != "")].copy()
    if df_with_layers.empty:
        return

    df_with_layers["num_layers"] = df_with_layers["num_layers"].astype(int)

    fig, ax = plt.subplots(figsize=(8, 5))

    for metric, marker, label in [("comet", "o", "COMET"),
                                   ("chrf", "s", "chrF++ (÷100)"),
                                   ("bleu", "^", "BLEU (÷100)")]:
        for _, row in df_with_layers.iterrows():
            val = row[metric]
            if pd.isna(val):
                continue
            # Normalize chrF++ and BLEU to [0,1] scale for comparison with COMET
            if metric in ("chrf", "bleu"):
                val = val / 100
            ax.scatter(row["num_layers"], val, c=color_for(row["experiment_id"]),
                       marker=marker, s=60, zorder=3)
            ax.annotate(row["experiment_id"],
                        (row["num_layers"], val),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

    # Marker legend
    from matplotlib.lines import Line2D
    marker_legend = [
        Line2D([0], [0], marker="o", color="gray", linestyle="", label="COMET"),
        Line2D([0], [0], marker="s", color="gray", linestyle="", label="chrF++ (÷100)"),
        Line2D([0], [0], marker="^", color="gray", linestyle="", label="BLEU (÷100)"),
    ]
    ax.legend(handles=marker_legend, fontsize=9, loc="lower right")

    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Score")
    ax.set_title("Translation Quality vs Number of Layers")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    fig.savefig(output_dir / "layers_vs_quality.png")
    plt.close(fig)
    print(f"  Saved layers_vs_quality.png")


def plot_group_summary(df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart comparing average metrics per experiment group."""
    df_grouped = df.copy()
    df_grouped["group"] = df_grouped["experiment_id"].apply(get_group)

    groups = df_grouped.groupby("group").agg({
        "comet": "mean",
        "chrf": "mean",
        "bleu": "mean",
        "size_mb": "mean",
    }).reindex(["B", "M", "I", "L"]).dropna(how="all")

    if len(groups) < 2:
        # Not enough groups for a meaningful comparison
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Quality comparison
    x = range(len(groups))
    width = 0.25
    colors_q = ["#4C72B0", "#DD8452", "#55A868"]
    for i, (metric, label) in enumerate([("comet", "COMET"),
                                          ("chrf", "chrF++ (÷10)"),
                                          ("bleu", "BLEU (÷10)")]):
        vals = groups[metric].tolist()
        if metric in ("chrf", "bleu"):
            vals = [v / 10 for v in vals]
        axes[0].bar([xi + i * width for xi in x], vals, width,
                    label=label, color=colors_q[i], edgecolor="white")

    axes[0].set_xticks([xi + width for xi in x])
    axes[0].set_xticklabels([GROUP_LABELS.get(g, g) for g in groups.index], fontsize=9)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Average Quality by Group")
    axes[0].legend(fontsize=9)

    # Size comparison
    axes[1].bar(range(len(groups)), groups["size_mb"].tolist(),
                color=[GROUP_COLORS.get(g, "#999") for g in groups.index],
                edgecolor="white")
    axes[1].set_xticks(range(len(groups)))
    axes[1].set_xticklabels([GROUP_LABELS.get(g, g) for g in groups.index], fontsize=9)
    axes[1].set_ylabel("Size (MB)")
    axes[1].set_title("Average Model Size by Group")

    plt.tight_layout()
    fig.savefig(output_dir / "group_summary.png")
    plt.close(fig)
    print(f"  Saved group_summary.png")


def print_summary_table(df: pd.DataFrame):
    """Print a formatted summary to terminal."""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    if df.empty:
        print("No results found.")
        return

    baseline = df[df["experiment_id"] == "B0"]

    for _, row in df.iterrows():
        eid = row["experiment_id"]
        desc = row.get("description", "")
        print(f"\n  {eid}: {desc}")
        print(f"    Layers: {row.get('num_layers', '?'):<6}  "
              f"Size: {row.get('size_mb', 0):,.0f} MB  "
              f"Speed: {row.get('tokens_per_sec', 0):.1f} tok/s")
        print(f"    COMET: {row.get('comet', 0):.4f}  "
              f"chrF++: {row.get('chrf', 0):.2f}  "
              f"BLEU: {row.get('bleu', 0):.2f}", end="")

        # Show delta from baseline
        if not baseline.empty and eid != "B0":
            b = baseline.iloc[0]
            dc = row.get("comet", 0) - b.get("comet", 0)
            dh = row.get("chrf", 0) - b.get("chrf", 0)
            db = row.get("bleu", 0) - b.get("bleu", 0)
            print(f"  (Δ {dc:+.4f} / {dh:+.2f} / {db:+.2f})", end="")
        print()

    print("\n" + "=" * 80)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--output", type=Path, default=Path("experiments/results/plots"),
                        help="Output directory for plots (default: experiments/results/plots/)")
    parser.add_argument("--groups", nargs="*", default=None,
                        help="Filter to specific groups (e.g., --groups I M)")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip terminal summary")
    args = parser.parse_args()

    df = collect_results()

    if df.empty:
        print("No results found in experiments/results/")
        sys.exit(1)

    # Filter groups if requested
    if args.groups:
        groups = [g.upper() for g in args.groups]
        df = df[df["experiment_id"].apply(get_group).isin(groups)]

    # Print summary
    if not args.no_show:
        print_summary_table(df)

    # Generate plots
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating plots in {args.output}/")

    plot_quality_metrics(df, args.output)
    plot_efficiency(df, args.output)
    plot_layers_vs_quality(df, args.output)
    plot_group_summary(df, args.output)

    print(f"\nDone! {len(df)} experiments plotted.")


if __name__ == "__main__":
    main()
