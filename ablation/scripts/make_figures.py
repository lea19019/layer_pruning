"""Generate all figures for the ablation study.

Reads results from ablation/results/ and produces publication-quality figures
in ablation/figures/.

Usage:
    python -m ablation.scripts.make_figures
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "ablation/results"
FIGURES_DIR = PROJECT_ROOT / "ablation/figures"

# Consistent styling
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
})

MODEL_COLORS = {
    "base": "#888888",
    "pruned_only": "#d62728",
    "pruned_ft_kd": "#2ca02c",
    "full_ft_kd": "#1f77b4",
}
MODEL_LABELS = {
    "base": "Base (unpruned, no FT)",
    "pruned_only": "Pruned only (no FT)",
    "pruned_ft_kd": "Pruned + FT + KD",
    "full_ft_kd": "Full + FT + KD (ceiling)",
}


def fig_ft_recovery_curves():
    """Figure 1: FT recovery curves — COMET over training steps for all 5 runs."""
    runs = {
        "full_kd": ("Full data + KD (165K)", "#1f77b4", "-"),
        "no_kd": ("Authentic only (100K)", "#2ca02c", "-"),
        "frac_0.75": ("75% data (124K)", "#ff7f0e", "--"),
        "frac_0.5": ("50% data (83K)", "#d62728", "--"),
        "frac_0.25": ("25% data (41K)", "#9467bd", "--"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Helper: extract step number from checkpoint name
    def ckpt_to_step(name, total_steps):
        if name.startswith("step_"):
            return int(name.split("_")[1])
        if name.startswith("epoch_"):
            ep = int(name.split("_")[1])
            return ep * total_steps // 3
        if name == "final":
            return total_steps
        return 0

    # Load training configs for total_steps
    total_steps_per_run = {}
    for run_name in runs:
        cfg_path = RESULTS_DIR / "ft_recovery" / run_name / "training_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                total_steps_per_run[run_name] = json.load(f).get("total_steps", 15000)
        else:
            total_steps_per_run[run_name] = 15000

    # Left: COMET recovery
    ax = axes[0]
    PRUNED_BASELINE_COMET = 0.315
    for run_name, (label, color, ls) in runs.items():
        path = RESULTS_DIR / "ft_recovery" / run_name / "recovery_curve.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        total = total_steps_per_run[run_name]
        steps = [ckpt_to_step(d["checkpoint"], total) for d in data]
        comets = [d["comet"] for d in data]
        # Sort by step
        pairs = sorted(zip(steps, comets))
        steps, comets = zip(*pairs)
        # Prepend step=0 at the pruned baseline so the curve shows the
        # dramatic recovery from 0.315 -> ~0.80 in the first 1000 steps.
        steps = (0,) + steps
        comets = (PRUNED_BASELINE_COMET,) + comets
        ax.plot(steps, comets, marker="o", color=color, linestyle=ls, label=label,
                markersize=5, linewidth=1.8)

    # Highlight the starting point shared by all runs
    ax.scatter([0], [PRUNED_BASELINE_COMET], s=120, facecolor="white",
               edgecolor="red", linewidth=2, zorder=5)
    ax.annotate(f"Pruned baseline\n(COMET={PRUNED_BASELINE_COMET:.3f})",
                xy=(0, PRUNED_BASELINE_COMET),
                xytext=(1800, 0.40),
                fontsize=9, color="red",
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.6, lw=1))

    # Reference ceilings
    ax.axhline(0.836, color="green", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(15200, 0.842, "I2_16 (0.836)", fontsize=9, color="green", alpha=0.7, ha="right")
    ax.axhline(0.893, color="blue", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(15200, 0.899, "B4 ceiling (0.893)", fontsize=9, color="blue", alpha=0.7, ha="right")

    ax.set_xlabel("Training step")
    ax.set_ylabel("COMET")
    ax.set_title("FT recovery curve: COMET vs. training steps")
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.25, 0.95)

    # Right: Peak COMET vs. data size
    ax = axes[1]
    data_sizes = {
        "frac_0.25": 41_000,
        "frac_0.5": 83_000,
        "frac_0.75": 124_000,
        "no_kd": 100_000,
        "full_kd": 165_000,
    }
    peaks = {}
    for run_name in runs:
        path = RESULTS_DIR / "ft_recovery" / run_name / "recovery_curve.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        peaks[run_name] = max(d["comet"] for d in data)

    # Separate: authentic-only scaling (frac runs + no_kd) vs KD comparison
    auth_runs = ["frac_0.25", "frac_0.5", "frac_0.75", "no_kd"]
    auth_sizes = [data_sizes[r] for r in auth_runs]
    auth_peaks = [peaks[r] for r in auth_runs]

    ax.plot(auth_sizes, auth_peaks, "o-", color="#2ca02c", markersize=10,
            linewidth=2, label="Authentic data only")
    ax.plot([data_sizes["full_kd"]], [peaks["full_kd"]], "s",
            color="#1f77b4", markersize=12, label="Authentic + KD")

    # Annotations
    for r, s, p in zip(auth_runs, auth_sizes, auth_peaks):
        ax.annotate(f"{p:.3f}", (s, p), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    ax.annotate(f"{peaks['full_kd']:.3f}", (data_sizes["full_kd"], peaks["full_kd"]),
                textcoords="offset points", xytext=(0, -18), ha="center", fontsize=9)

    ax.axhline(0.893, color="blue", linestyle=":", alpha=0.5, linewidth=1,
               label="B4 ceiling")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Peak COMET")
    ax.set_title("Peak COMET vs. training data size")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.7, 0.93)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ft_recovery_curves.png")
    plt.close()
    print("Saved ft_recovery_curves.png")


def fig_hidden_state_divergence():
    """Figure 2: CKA heatmaps showing divergence across models."""
    comparisons = [
        ("base_vs_pruned_only", "Base vs. Pruned only"),
        ("base_vs_pruned_ft_kd", "Base vs. Pruned+FT+KD"),
        ("base_vs_full_ft_kd", "Base vs. Full+FT+KD"),
        ("pruned_only_vs_pruned_ft_kd", "Pruned only vs. Pruned+FT+KD"),
        ("pruned_only_vs_full_ft_kd", "Pruned only vs. Full+FT+KD"),
        ("pruned_ft_kd_vs_full_ft_kd", "Pruned+FT+KD vs. Full+FT+KD"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, comparisons):
        cka_path = RESULTS_DIR / f"cka_{key}.npy"
        if not cka_path.exists():
            continue
        cka = np.load(cka_path)
        im = ax.imshow(cka, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0,
                       origin="lower")
        ax.set_title(title, fontsize=11)
        # Labels based on shape
        if cka.shape[0] == 32:
            ax.set_ylabel("Base layer")
        else:
            ax.set_ylabel("Pruned layer (0-15)")
        if cka.shape[1] == 32:
            ax.set_xlabel("Full model layer")
        else:
            ax.set_xlabel("Pruned layer (0-15)")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("Cross-Model CKA: Representation Similarity Across Layers",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hidden_state_cka_heatmaps.png")
    plt.close()
    print("Saved hidden_state_cka_heatmaps.png")


def fig_matched_depth_cka():
    """Figure 3: Matched-depth CKA — how similar pruned and base layers are."""
    with open(RESULTS_DIR / "hidden_state_divergence.json") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 5))

    for key, label, color in [
        ("matched_pruned_only_vs_base", "Pruned vs. Base", "#d62728"),
        ("matched_pruned_only_vs_full_ft_kd", "Pruned vs. Full+FT+KD", "#888888"),
        ("matched_pruned_ft_kd_vs_base", "Pruned+FT+KD vs. Base", "#ff7f0e"),
        ("matched_pruned_ft_kd_vs_full_ft_kd", "Pruned+FT+KD vs. Full+FT+KD", "#1f77b4"),
    ]:
        if key not in data:
            continue
        rel_depths = [m["rel_depth"] for m in data[key]]
        ckas = [m["cka"] for m in data[key]]
        ax.plot(rel_depths, ckas, "o-", color=color, label=label, markersize=6, linewidth=1.8)

    ax.set_xlabel("Relative depth (0 = input, 1 = output)")
    ax.set_ylabel("CKA")
    ax.set_title("Matched-Depth CKA: Pruned (16 layers) vs. Full (32 layers)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.05)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "matched_depth_cka.png")
    plt.close()
    print("Saved matched_depth_cka.png")


def fig_redundancy_analysis():
    """Figure 4: Pairwise CKA within base vs full+FT+KD + effective rank."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top row: pairwise CKA heatmaps
    for ax, name in zip(axes[0], ["base", "full_ft_kd"]):
        cka = np.load(RESULTS_DIR / f"pairwise_cka_{name}.npy")
        im = ax.imshow(cka, cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
        ax.set_title(f"Pairwise CKA within {MODEL_LABELS[name]}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Bottom left: adjacent-layer CKA
    ax = axes[1, 0]
    with open(RESULTS_DIR / "redundancy_analysis.json") as f:
        redun = json.load(f)
    for name in ["base", "full_ft_kd"]:
        if name not in redun:
            continue
        adj = redun[name]["adjacent_cka"]
        ax.plot(range(len(adj)), adj, "o-", color=MODEL_COLORS[name],
                label=MODEL_LABELS[name], markersize=5, linewidth=1.8)

    # Show which layers were pruned
    pruned_layers = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23]
    for layer in pruned_layers:
        ax.axvline(layer - 0.5, color="red", alpha=0.08)
    ax.text(11, 0.76, "Layers removed by IFR", color="red", alpha=0.7,
            ha="center", fontsize=9)

    ax.set_xlabel("Layer i (pair: i vs. i+1)")
    ax.set_ylabel("CKA(layer_i, layer_{i+1})")
    ax.set_title("Adjacent-Layer CKA (redundancy indicator)")
    ax.axhline(0.95, color="black", linestyle=":", alpha=0.4, label="Redundancy threshold (0.95)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Bottom right: effective rank
    ax = axes[1, 1]
    for name in ["base", "full_ft_kd"]:
        erank_path = RESULTS_DIR / f"effective_rank_{name}.npy"
        if not erank_path.exists():
            continue
        erank = np.load(erank_path)
        ax.plot(range(len(erank)), erank, "o-", color=MODEL_COLORS[name],
                label=MODEL_LABELS[name], markersize=5, linewidth=1.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective rank")
    ax.set_title("Effective Rank per Layer (representational richness)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "redundancy_analysis.png")
    plt.close()
    print("Saved redundancy_analysis.png")


def fig_logit_lens():
    """Figure 5: Logit lens — entropy and target rank across layers."""
    with open(RESULTS_DIR / "logit_lens.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: entropy per layer
    ax = axes[0]
    for name in ["base", "pruned_only", "pruned_ft_kd", "full_ft_kd"]:
        if name not in data:
            continue
        entropies = data[name]["avg_entropy_per_layer"]
        rel_depth = np.linspace(0, 1, len(entropies))
        ax.plot(rel_depth, entropies, "o-", color=MODEL_COLORS[name],
                label=MODEL_LABELS[name], markersize=5, linewidth=1.8)
    ax.set_xlabel("Relative depth")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Output distribution entropy per layer")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    # Right: target rank per layer
    ax = axes[1]
    for name in ["base", "pruned_only", "pruned_ft_kd", "full_ft_kd"]:
        if name not in data:
            continue
        ranks = data[name]["avg_target_rank_per_layer"]
        ranks = [r for r in ranks if r >= 0]  # filter -1 sentinels
        if not ranks:
            continue
        rel_depth = np.linspace(0, 1, len(ranks))
        ax.plot(rel_depth, ranks, "o-", color=MODEL_COLORS[name],
                label=MODEL_LABELS[name], markersize=5, linewidth=1.8)
    ax.set_xlabel("Relative depth")
    ax.set_ylabel("Avg rank of target token (log scale)")
    ax.set_yscale("log")
    ax.set_title("Target token rank per layer (lower = model 'knows' the answer)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "logit_lens.png")
    plt.close()
    print("Saved logit_lens.png")


def fig_attention_comparison():
    """Figure 6: Attention entropy and BOS attention across layers."""
    with open(RESULTS_DIR / "attention_comparison.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: attention entropy per layer
    ax = axes[0]
    for name in ["pruned_only", "pruned_ft_kd", "full_ft_kd"]:
        if name not in data:
            continue
        ent = data[name]["avg_entropy_per_layer"]
        rel_depth = np.linspace(0, 1, len(ent))
        ax.plot(rel_depth, ent, "o-", color=MODEL_COLORS[name],
                label=MODEL_LABELS[name], markersize=5, linewidth=1.8)
    ax.set_xlabel("Relative depth")
    ax.set_ylabel("Attention entropy")
    ax.set_title("Attention distribution entropy (focus vs. spread)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    # Right: BOS attention fraction
    ax = axes[1]
    for name in ["pruned_only", "pruned_ft_kd", "full_ft_kd"]:
        if name not in data:
            continue
        bos = data[name]["avg_bos_frac_per_layer"]
        rel_depth = np.linspace(0, 1, len(bos))
        ax.plot(rel_depth, bos, "o-", color=MODEL_COLORS[name],
                label=MODEL_LABELS[name], markersize=5, linewidth=1.8)
    ax.set_xlabel("Relative depth")
    ax.set_ylabel("Fraction of attention on BOS")
    ax.set_title("BOS-token attention (attention-sink behavior)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "attention_comparison.png")
    plt.close()
    print("Saved attention_comparison.png")


def fig_weight_diff():
    """Figure 7: Per-layer weight change during FT."""
    with open(RESULTS_DIR / "weight_diff_per_layer.json") as f:
        data = json.load(f)

    layers = [d["layer"] for d in data]
    rel_change = [d["rel_change"] for d in data]
    attn_diff = [d["attn_diff"] for d in data]
    mlp_diff = [d["mlp_diff"] for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: relative change per layer
    ax = axes[0]
    bars = ax.bar(layers, rel_change, color="#1f77b4", alpha=0.8)
    # Color top 3 changes
    top3 = sorted(range(len(rel_change)), key=lambda i: -rel_change[i])[:3]
    for i in top3:
        bars[i].set_color("#d62728")
    ax.set_xlabel("Layer (in pruned model, 0-15)")
    ax.set_ylabel("Relative weight change ||ΔW|| / ||W||")
    ax.set_title("Per-layer weight change during FT (IP_16 → I2_16)")
    ax.grid(alpha=0.3, axis="y")

    # Right: attn vs mlp change
    ax = axes[1]
    x = np.arange(len(layers))
    width = 0.4
    ax.bar(x - width/2, attn_diff, width, color="#ff7f0e", label="Attention ||ΔW||")
    ax.bar(x + width/2, mlp_diff, width, color="#2ca02c", label="MLP ||ΔW||")
    ax.set_xlabel("Layer (in pruned model, 0-15)")
    ax.set_ylabel("||ΔW|| (Frobenius norm)")
    ax.set_title("Attention vs. MLP weight change during FT")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "weight_diff_per_layer.png")
    plt.close()
    print("Saved weight_diff_per_layer.png")


def fig_output_categorization():
    """Figure 8: Error category distribution across models."""
    with open(RESULTS_DIR / "output_categorization.json") as f:
        data = json.load(f)

    models = ["IP_16_enes", "I2_16_enes", "B4_enes"]
    labels = ["Pruned only", "Pruned+FT+KD", "Full+FT+KD"]

    # Collect all categories across models
    all_cats = set()
    for m in models:
        if m in data:
            all_cats.update(data[m]["summary"]["category_counts"].keys())
    all_cats = sorted(all_cats)

    # Build matrix: categories x models
    category_order = ["plausible", "wrong_language", "verbose", "repetition",
                      "truncation", "source_copy", "garbage", "unclear"]
    ordered_cats = [c for c in category_order if c in all_cats] + \
                   [c for c in all_cats if c not in category_order]

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.25
    x = np.arange(len(ordered_cats))
    colors = ["#d62728", "#2ca02c", "#1f77b4"]

    for i, (m, lbl) in enumerate(zip(models, labels)):
        if m not in data:
            continue
        pcts = [data[m]["summary"]["category_pcts"].get(c, 0) for c in ordered_cats]
        ax.bar(x + (i - 1) * bar_width, pcts, bar_width, label=lbl, color=colors[i])

    ax.set_xlabel("Error category")
    ax.set_ylabel("% of translations")
    ax.set_title("Output error categories across models (10 sample translations)")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_cats, rotation=30, ha="right")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "output_categorization.png")
    plt.close()
    print("Saved output_categorization.png")


def fig_summary_comparison():
    """Figure 9: One-panel summary comparing all models and experiments."""
    models = ["base", "pruned_only", "pruned_ft_kd", "full_ft_kd"]
    comets = {"base": 0.582, "pruned_only": 0.315, "pruned_ft_kd": 0.836, "full_ft_kd": 0.893}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: COMET comparison
    ax = axes[0]
    values = [comets[m] for m in models]
    colors = [MODEL_COLORS[m] for m in models]
    labels = [MODEL_LABELS[m].replace(" (ceiling)", "") for m in models]
    bars = ax.bar(range(len(models)), values, color=colors, alpha=0.85)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("COMET")
    ax.set_title("Original experiments: COMET across all 4 ablation models")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")

    # Right: FT recovery peak COMET per run
    ax = axes[1]
    run_names = ["frac_0.25", "frac_0.5", "frac_0.75", "no_kd", "full_kd"]
    run_labels = ["25% data\n(no KD)", "50% data\n(no KD)", "75% data\n(no KD)",
                  "100% data\n(no KD)", "100% + KD\n(165K)"]
    peaks = []
    for r in run_names:
        path = RESULTS_DIR / "ft_recovery" / r / "recovery_curve.json"
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            peaks.append(max(x["comet"] for x in d))
        else:
            peaks.append(0)

    colors2 = ["#9467bd", "#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
    bars = ax.bar(range(len(run_names)), peaks, color=colors2, alpha=0.85)
    for bar, p in zip(bars, peaks):
        ax.text(bar.get_x() + bar.get_width()/2, p + 0.005, f"{p:.3f}",
                ha="center", fontsize=10, fontweight="bold")

    # Reference lines
    ax.axhline(0.836, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(4.3, 0.838, "I2_16 (0.836)", fontsize=8, color="gray", alpha=0.7)
    ax.axhline(0.893, color="blue", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(4.3, 0.895, "B4 (0.893)", fontsize=8, color="blue", alpha=0.7)

    ax.set_ylabel("Peak COMET")
    ax.set_title("FT recovery experiments: peak COMET per run")
    ax.set_xticks(range(len(run_names)))
    ax.set_xticklabels(run_labels)
    ax.set_ylim(0.7, 0.92)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "summary_comparison.png")
    plt.close()
    print("Saved summary_comparison.png")


def fig_surgical_fix_comparison():
    """Figure: COMET of surgical approaches vs. full LoRA FT."""
    path = RESULTS_DIR / "surgical_fix/surgical_all.json"
    if not path.exists():
        print("surgical_all.json not found; skipping surgical_fix_comparison")
        return
    with open(path) as f:
        surgical = json.load(f)

    # Reference scores
    pruned_baseline = 0.315
    full_lora_no_kd = 0.847

    # Build lists
    labels = ["Pruned\nbaseline"]
    comets = [pruned_baseline]
    colors = ["#888888"]
    for r in surgical:
        labels.append(r["approach"].replace("_", "\n"))
        comets.append(r["comet"])
        colors.append("#d62728")  # red for failed approaches
    labels.append("Full LoRA\n(no_kd)")
    comets.append(full_lora_no_kd)
    colors.append("#2ca02c")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(labels)), comets, color=colors, alpha=0.85)
    for bar, v in zip(bars, comets):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("COMET")
    ax.set_title("Surgical fix approaches vs. Full LoRA FT")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.axhline(pruned_baseline, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(full_lora_no_kd, color="green", linestyle=":", alpha=0.5)
    ax.grid(alpha=0.3, axis="y")

    # Annotate trainable params
    details = {
        "norm_rescale": "1 scalar\n(no training)",
        "linear_probe": "~17M params\n(closed-form)",
        "lm_head_ft": "1.05B params\n(5K ex, 1 epoch)",
        "mlp_last_ft": "528M params\n(5K ex, 1 epoch)",
    }
    for i, r in enumerate(surgical):
        name = r["approach"]
        if name in details:
            ax.text(i + 1, 0.02, details[name], ha="center", fontsize=8,
                    color="#555")
    ax.text(len(labels) - 1, 0.02, "21M params\n(100K ex, 3 ep)\nLoRA r=16 all layers",
            ha="center", fontsize=8, color="#225522")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "surgical_fix_comparison.png")
    plt.close()
    print("Saved surgical_fix_comparison.png")


def fig_all_surgical_comparison():
    """Combined figure: all 8 surgical attempts (v1 + v2) plus LoRA ceiling."""
    v1_path = RESULTS_DIR / "surgical_fix/surgical_all.json"
    v2_path = RESULTS_DIR / "surgical_fix_v2/surgical_v2_all.json"
    if not v1_path.exists() or not v2_path.exists():
        print("missing surgical results; skipping fig_all_surgical_comparison")
        return

    with open(v1_path) as f:
        v1 = json.load(f)
    with open(v2_path) as f:
        v2 = json.load(f)

    pruned_baseline = 0.315
    full_lora = 0.847

    # Order: baseline, v1 (localized), v2 (distributed), LoRA
    labels = ["Pruned\nbaseline"]
    comets = [pruned_baseline]
    categories = ["ref"]

    v1_order = ["norm_rescale", "linear_probe", "lm_head_ft", "mlp_last_ft"]
    v1_display = {
        "norm_rescale": "Norm\nrescale",
        "linear_probe": "Linear\nprobe",
        "lm_head_ft": "LM head\nFT",
        "mlp_last_ft": "Last-3 MLP\nFT",
    }
    v1_dict = {r["approach"]: r for r in v1}
    for a in v1_order:
        labels.append(v1_display[a])
        comets.append(v1_dict[a]["comet"])
        categories.append("v1")

    v2_order = ["per_layer_norm", "procrustes", "low_rank_probes", "bias_only"]
    v2_display = {
        "per_layer_norm": "Per-layer\nnorm",
        "procrustes": "Procrustes\n(rotations)",
        "low_rank_probes": "Low-rank\nprobes r=16",
        "bias_only": "Bias-only\nFT",
    }
    v2_dict = {r["approach"]: r for r in v2}
    for a in v2_order:
        if a in v2_dict:
            labels.append(v2_display[a])
            comets.append(v2_dict[a]["comet"])
            categories.append("v2")

    labels.append("Full LoRA\n(no_kd)")
    comets.append(full_lora)
    categories.append("lora")

    cat_colors = {
        "ref": "#888888",
        "v1": "#d62728",      # localized failures (red)
        "v2": "#ff7f0e",      # distributed attempts (orange)
        "lora": "#2ca02c",    # ceiling (green)
    }
    colors = [cat_colors[c] for c in categories]

    fig, ax = plt.subplots(figsize=(15, 6.5))
    bars = ax.bar(range(len(labels)), comets, color=colors, alpha=0.85)
    for bar, v in zip(bars, comets):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("COMET")
    ax.set_title("All surgical attempts: localized (red) → distributed (orange) → full LoRA (green)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.axhline(pruned_baseline, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(full_lora, color="green", linestyle=":", alpha=0.5)
    ax.grid(alpha=0.3, axis="y")

    # Legend via proxy artists
    import matplotlib.patches as mpatches
    legend_elems = [
        mpatches.Patch(color="#888888", label="Reference"),
        mpatches.Patch(color="#d62728", label="v1: localized (one interface point)"),
        mpatches.Patch(color="#ff7f0e", label="v2: distributed (every layer)"),
        mpatches.Patch(color="#2ca02c", label="Full LoRA ceiling"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=9)

    # Draw separator lines
    ax.axvline(0.5, color="black", alpha=0.15, linestyle="-", linewidth=0.5)
    ax.axvline(4.5, color="black", alpha=0.15, linestyle="-", linewidth=0.5)
    ax.axvline(8.5, color="black", alpha=0.15, linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "all_surgical_comparison.png")
    plt.close()
    print("Saved all_surgical_comparison.png")


def fig_procrustes_error_by_layer():
    """Figure: Procrustes reconstruction error grows with depth."""
    v2_path = RESULTS_DIR / "surgical_fix_v2/surgical_v2_all.json"
    if not v2_path.exists():
        return
    with open(v2_path) as f:
        v2 = json.load(f)

    proc = next((r for r in v2 if r["approach"] == "procrustes"), None)
    lrp = next((r for r in v2 if r["approach"] == "low_rank_probes"), None)
    if not proc:
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))
    errors = proc["reconstruction_errors"]
    layers = list(range(len(errors)))
    ax.plot(layers, errors, "o-", color="#ff7f0e", markersize=7, linewidth=2,
            label="Procrustes (orthogonal rotation only)")

    if lrp:
        lrp_errors = lrp["reconstruction_errors"]
        ax.plot(layers, lrp_errors, "s-", color="#1f77b4", markersize=7, linewidth=2,
                label=f"Low-rank probes r={lrp['rank']} (closed-form)")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Relative reconstruction error")
    ax.set_title("Per-layer residual correction error: depth-compounding")
    ax.legend()
    ax.grid(alpha=0.3)

    # Highlight the growth for procrustes
    ax.annotate(f"Error grows {errors[0]:.2f} → {errors[-1]:.2f}\n(5.6× increase)",
                xy=(len(errors) - 1, errors[-1]),
                xytext=(len(errors) - 6, 0.3),
                fontsize=10, color="#ff7f0e",
                arrowprops=dict(arrowstyle="->", color="#ff7f0e", alpha=0.6))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "procrustes_error_by_layer.png")
    plt.close()
    print("Saved procrustes_error_by_layer.png")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {FIGURES_DIR}\n")

    fig_ft_recovery_curves()
    fig_hidden_state_divergence()
    fig_matched_depth_cka()
    fig_redundancy_analysis()
    fig_logit_lens()
    fig_attention_comparison()
    fig_weight_diff()
    fig_output_categorization()
    fig_summary_comparison()
    fig_surgical_fix_comparison()
    fig_all_surgical_comparison()
    fig_procrustes_error_by_layer()

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
