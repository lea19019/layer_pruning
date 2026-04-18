"""Experiment 8: Weight diff analysis between pruned and pruned+FT models.

Compares LoRA-modified weights between the pruned-only model and the
pruned+FT+KD model to understand which layers changed most during FT.

This is CPU-only — compares saved model weights on disk without loading
to GPU.

Metrics:
  - Per-layer Frobenius norm of weight difference
  - Per-layer relative change (||W_ft - W_pruned|| / ||W_pruned||)
  - Per-module breakdown (attention vs MLP)
  - Which layers absorbed most change during FT

Usage:
    python -m ablation.scripts.weight_diff_analysis
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "ablation/results"

PRUNED_MODEL = PROJECT_ROOT / "experiments/results/IP_16_enes/pruned_model"
FT_MODEL = PROJECT_ROOT / "experiments/results/I2_16_enes/finetuned/merged"


def load_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load model state dict from safetensors."""
    safetensor_files = list(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files in {model_dir}")

    state_dict = {}
    for f in safetensor_files:
        state_dict.update(load_file(str(f), device="cpu"))
    return state_dict


def extract_layer_info(key: str) -> tuple[int | None, str]:
    """Extract layer index and module type from parameter name.

    Returns (layer_idx, module_type) where module_type is one of:
    'attn', 'mlp', 'embed', 'head', 'norm', 'other'
    """
    parts = key.split(".")

    layer_idx = None
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            layer_idx = int(parts[i + 1])
            break

    if "self_attn" in key or any(p in key for p in ["q_proj", "k_proj", "v_proj", "o_proj"]):
        module_type = "attn"
    elif "mlp" in key or any(p in key for p in ["gate_proj", "up_proj", "down_proj"]):
        module_type = "mlp"
    elif "embed" in key:
        module_type = "embed"
    elif "lm_head" in key:
        module_type = "head"
    elif "norm" in key:
        module_type = "norm"
    else:
        module_type = "other"

    return layer_idx, module_type


def compute_weight_diffs(
    pruned_dict: dict[str, torch.Tensor],
    ft_dict: dict[str, torch.Tensor],
) -> dict:
    """Compute per-parameter and per-layer weight differences."""
    common_keys = set(pruned_dict.keys()) & set(ft_dict.keys())
    print(f"Common parameters: {len(common_keys)}")

    per_param = []
    per_layer = {}
    per_module = {"attn": 0.0, "mlp": 0.0, "embed": 0.0, "head": 0.0, "norm": 0.0, "other": 0.0}
    per_module_baseline = {"attn": 0.0, "mlp": 0.0, "embed": 0.0, "head": 0.0, "norm": 0.0, "other": 0.0}

    for key in sorted(common_keys):
        w_pruned = pruned_dict[key].float()
        w_ft = ft_dict[key].float()

        if w_pruned.shape != w_ft.shape:
            continue

        diff_norm = torch.norm(w_ft - w_pruned).item()
        base_norm = torch.norm(w_pruned).item()
        rel_change = diff_norm / (base_norm + 1e-10)

        layer_idx, module_type = extract_layer_info(key)

        per_param.append({
            "param": key,
            "layer": layer_idx,
            "module": module_type,
            "diff_norm": round(diff_norm, 6),
            "base_norm": round(base_norm, 6),
            "rel_change": round(rel_change, 6),
            "shape": list(w_pruned.shape),
        })

        per_module[module_type] += diff_norm
        per_module_baseline[module_type] += base_norm

        if layer_idx is not None:
            if layer_idx not in per_layer:
                per_layer[layer_idx] = {
                    "attn_diff": 0.0, "mlp_diff": 0.0, "norm_diff": 0.0,
                    "attn_base": 0.0, "mlp_base": 0.0, "norm_base": 0.0,
                    "total_diff": 0.0, "total_base": 0.0,
                }
            per_layer[layer_idx][f"{module_type}_diff"] += diff_norm
            per_layer[layer_idx][f"{module_type}_base"] += base_norm
            per_layer[layer_idx]["total_diff"] += diff_norm
            per_layer[layer_idx]["total_base"] += base_norm

    # Compute per-layer relative change
    for layer_idx in per_layer:
        d = per_layer[layer_idx]
        d["rel_change"] = round(
            d["total_diff"] / (d["total_base"] + 1e-10), 6
        )
        for key in list(d.keys()):
            if isinstance(d[key], float):
                d[key] = round(d[key], 6)

    # Module-level relative change
    per_module_rel = {
        k: round(per_module[k] / (per_module_baseline[k] + 1e-10), 6)
        for k in per_module
    }

    return {
        "per_param": per_param,
        "per_layer": {str(k): v for k, v in sorted(per_layer.items())},
        "per_module_total_diff": {k: round(v, 4) for k, v in per_module.items()},
        "per_module_rel_change": per_module_rel,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading pruned model weights...")
    pruned_dict = load_state_dict(PRUNED_MODEL)
    print(f"  Parameters: {len(pruned_dict)}")

    print("Loading fine-tuned model weights...")
    ft_dict = load_state_dict(FT_MODEL)
    print(f"  Parameters: {len(ft_dict)}")

    print("\nComputing weight differences...")
    results = compute_weight_diffs(pruned_dict, ft_dict)

    # Print summary
    print("\n=== Per-Layer Relative Change ===")
    for layer_str, info in sorted(results["per_layer"].items(), key=lambda x: int(x[0])):
        rel = info["rel_change"]
        bar = "#" * int(rel * 200)
        print(f"  Layer {layer_str:>2}: {rel:.4f} {bar}")

    print("\n=== Per-Module Relative Change ===")
    for module, rel in results["per_module_rel_change"].items():
        print(f"  {module:>6}: {rel:.4f}")

    # Top 10 most-changed parameters
    top_params = sorted(results["per_param"], key=lambda x: -x["rel_change"])[:10]
    print("\n=== Top 10 Most-Changed Parameters ===")
    for p in top_params:
        print(f"  {p['param']}: rel_change={p['rel_change']:.4f}")

    # Save
    with open(RESULTS_DIR / "weight_diff_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/weight_diff_analysis.json")

    # Also save a compact per-layer summary
    layer_summary = []
    for layer_str, info in sorted(results["per_layer"].items(), key=lambda x: int(x[0])):
        layer_summary.append({
            "layer": int(layer_str),
            "rel_change": info["rel_change"],
            "attn_diff": info["attn_diff"],
            "mlp_diff": info["mlp_diff"],
        })
    with open(RESULTS_DIR / "weight_diff_per_layer.json", "w") as f:
        json.dump(layer_summary, f, indent=2)


if __name__ == "__main__":
    main()
