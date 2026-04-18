"""Experiment 4: Attention pattern comparison across models.

Compares attention patterns in the surviving layers (0-5 and 24-31) between
pruned-only and pruned+FT models. Do attention heads reorganize after FT
to compensate for missing middle layers?

Metrics per head:
  - Average attention entropy (spread vs. focused)
  - Fraction of attention on source tokens vs. target prefix
  - Attention to specific positions (BOS, last source token, etc.)

Usage:
    python -m ablation.scripts.attention_comparison [--n-samples 50]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "ablation/results"

MODELS = {
    "pruned_only": str(PROJECT_ROOT / "experiments/results/IP_16_enes/pruned_model"),
    "pruned_ft_kd": str(
        PROJECT_ROOT / "experiments/results/I2_16_enes/finetuned/merged"
    ),
    "full_ft_kd": str(PROJECT_ROOT / "experiments/results/B4_enes/finetuned/merged"),
}

TEST_EN = PROJECT_ROOT / "data/filtered_en_es/test.en"


def load_model(path: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def collect_attention_weights(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> list[np.ndarray]:
    """Extract attention weights from all layers.

    Returns list of (n_heads, seq_len, seq_len) numpy arrays, one per layer.
    """
    # Cohere uses SDPA/flash attention by default which doesn't return weights.
    # Force eager attention implementation so we can capture attention patterns.
    # Also set output_attentions=True on each attention module's config.
    original_attn_impl = getattr(model.config, "_attn_implementation", "eager")
    model.config._attn_implementation = "eager"
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "config"):
            layer.self_attn.config._attn_implementation = "eager"

    attn_weights = {}
    hooks = []
    layers = model.model.layers

    for i, layer in enumerate(layers):
        def make_hook(idx):
            def hook(module, input, output):
                # Cohere attention returns (attn_output, attn_weights, past_kv)
                if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                    attn_weights[idx] = output[1].detach().cpu().float().numpy()
            return hook
        hooks.append(layer.self_attn.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        kwargs = {"input_ids": input_ids, "output_attentions": True}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        model(**kwargs)

    # Restore original attention implementation
    model.config._attn_implementation = original_attn_impl

    for h in hooks:
        h.remove()

    n_layers = len(layers)
    results = []
    for i in range(n_layers):
        if i in attn_weights:
            # (batch, n_heads, seq, seq) -> take first batch element
            results.append(attn_weights[i][0])
        else:
            results.append(None)
    return results


def attention_entropy(attn: np.ndarray) -> float:
    """Compute average entropy of attention distributions.

    Args:
        attn: (n_heads, seq_len, seq_len) attention weights.

    Returns:
        Average entropy across heads and query positions.
    """
    # Clip to avoid log(0)
    attn_clipped = np.clip(attn, 1e-10, 1.0)
    entropy = -np.sum(attn_clipped * np.log(attn_clipped), axis=-1)
    return float(entropy.mean())


def attention_concentration(attn: np.ndarray) -> dict:
    """Measure how concentrated attention is on specific positions.

    Returns:
        Dict with fraction of attention on BOS, last position, diagonal.
    """
    n_heads, seq_len, _ = attn.shape
    if seq_len == 0:
        return {"bos_frac": 0.0, "last_frac": 0.0, "diag_frac": 0.0}

    bos_frac = float(attn[:, :, 0].mean())
    last_query_attn = attn[:, -1, :]  # (n_heads, seq_len)
    # Fraction on last few positions (the model's own recent context)
    recent_window = min(5, seq_len)
    recent_frac = float(last_query_attn[:, -recent_window:].sum(axis=-1).mean())

    # Diagonal attention (token attending to itself)
    diag_vals = np.array([attn[:, i, i].mean() for i in range(seq_len)])
    diag_frac = float(diag_vals.mean())

    return {
        "bos_frac": round(bos_frac, 4),
        "recent_frac": round(recent_frac, 4),
        "diag_frac": round(diag_frac, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(TEST_EN) as f:
        sources = [line.strip() for line in f if line.strip()][:args.n_samples]

    prompts = [f"Translate the following English sentence to Spanish: {s}" for s in sources]

    all_results = {}

    for name, path in MODELS.items():
        print(f"\n=== {name} ===")
        model, tokenizer = load_model(path, args.device)
        n_layers = model.config.num_hidden_layers

        # Aggregate per-layer stats
        layer_entropies = np.zeros(n_layers)
        layer_bos_frac = np.zeros(n_layers)
        layer_recent_frac = np.zeros(n_layers)
        layer_diag_frac = np.zeros(n_layers)
        count = 0

        for prompt in prompts:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256
            )
            inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

            attn_list = collect_attention_weights(
                model, inputs["input_ids"], inputs.get("attention_mask")
            )

            for li, attn in enumerate(attn_list):
                if attn is None:
                    continue
                layer_entropies[li] += attention_entropy(attn)
                conc = attention_concentration(attn)
                layer_bos_frac[li] += conc["bos_frac"]
                layer_recent_frac[li] += conc["recent_frac"]
                layer_diag_frac[li] += conc["diag_frac"]
            count += 1

        layer_entropies /= max(count, 1)
        layer_bos_frac /= max(count, 1)
        layer_recent_frac /= max(count, 1)
        layer_diag_frac /= max(count, 1)

        all_results[name] = {
            "n_layers": n_layers,
            "n_samples": count,
            "avg_entropy_per_layer": layer_entropies.tolist(),
            "avg_bos_frac_per_layer": layer_bos_frac.tolist(),
            "avg_recent_frac_per_layer": layer_recent_frac.tolist(),
            "avg_diag_frac_per_layer": layer_diag_frac.tolist(),
        }

        print(f"  Layers: {n_layers}")
        print(f"  Avg entropy: {layer_entropies.mean():.3f}")
        print(f"  Avg BOS attention: {layer_bos_frac.mean():.4f}")

        del model
        torch.cuda.empty_cache()

    with open(RESULTS_DIR / "attention_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    np.savez(
        RESULTS_DIR / "attention_comparison.npz",
        **{f"{name}_entropy": np.array(r["avg_entropy_per_layer"])
           for name, r in all_results.items()},
        **{f"{name}_bos": np.array(r["avg_bos_frac_per_layer"])
           for name, r in all_results.items()},
    )
    print(f"\nResults saved to {RESULTS_DIR}/attention_comparison.*")


if __name__ == "__main__":
    main()
