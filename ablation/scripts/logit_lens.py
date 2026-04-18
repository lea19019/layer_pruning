"""Experiment 3: Logit lens on pruned model.

Applies the language model head at each intermediate layer to see when the
model "knows" it should output Spanish. Compares across all 4 ablation models.

For each layer, we project the residual stream through the LM head and check:
  - Top-1 token prediction at each layer
  - Whether the predicted token is in Spanish vocabulary
  - Rank of the correct target token at each layer
  - Entropy of the output distribution (confidence)

Usage:
    python -m ablation.scripts.logit_lens [--n-samples 50] [--batch-size 4]
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
    "base": "CohereForAI/aya-expanse-8b",
    "pruned_only": str(PROJECT_ROOT / "experiments/results/IP_16_enes/pruned_model"),
    "pruned_ft_kd": str(
        PROJECT_ROOT / "experiments/results/I2_16_enes/finetuned/merged"
    ),
    "full_ft_kd": str(PROJECT_ROOT / "experiments/results/B4_enes/finetuned/merged"),
}

TEST_EN = PROJECT_ROOT / "data/filtered_en_es/test.en"
TEST_ES = PROJECT_ROOT / "data/filtered_en_es/test.es"


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


def logit_lens_single(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    target_prefix: str = "",
) -> dict:
    """Apply logit lens at each layer for a single prompt.

    Returns per-layer: top-5 predictions, entropy, and target rank (if provided).
    """
    activations = {}
    hooks = []
    layers = model.model.layers

    for i, layer in enumerate(layers):
        def make_hook(idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                activations[idx] = hidden.detach()
            return hook
        hooks.append(layer.register_forward_hook(make_hook(i)))

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    # Get the LM head (and any final layer norm)
    lm_head = model.lm_head
    if hasattr(model.model, "norm"):
        final_norm = model.model.norm
    else:
        final_norm = None

    # Encode target prefix for rank checking
    target_ids = None
    if target_prefix:
        target_ids = tokenizer.encode(target_prefix, add_special_tokens=False)

    n_layers = len(layers)
    layer_results = []

    for i in range(n_layers):
        hidden = activations[i][:, -1, :]  # last token, (1, hidden_dim)

        # Apply final norm if it exists
        if final_norm is not None:
            hidden = final_norm(hidden)

        # Cast both hidden and lm_head to float32 to avoid dtype mismatch
        # (model loaded in bfloat16, but we need float for softmax precision)
        logits = torch.nn.functional.linear(
            hidden.float(), lm_head.weight.float(),
            lm_head.bias.float() if lm_head.bias is not None else None
        )[0]  # (vocab_size,)
        probs = torch.softmax(logits, dim=-1)

        # Entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

        # Top-5 predictions
        top5_probs, top5_ids = torch.topk(probs, 5)
        top5_tokens = [tokenizer.decode([tid]) for tid in top5_ids.tolist()]

        # Target rank
        target_rank = None
        if target_ids:
            first_target = target_ids[0]
            sorted_ids = torch.argsort(logits, descending=True)
            target_rank = (sorted_ids == first_target).nonzero(as_tuple=True)[0].item()

        layer_results.append({
            "layer": i,
            "rel_depth": round(i / max(n_layers - 1, 1), 3),
            "top5_tokens": top5_tokens,
            "top5_probs": [round(p, 4) for p in top5_probs.tolist()],
            "entropy": round(entropy, 3),
            "target_rank": target_rank,
        })

    return {"n_layers": n_layers, "layers": layer_results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(TEST_EN) as f:
        sources = [line.strip() for line in f if line.strip()][:args.n_samples]
    with open(TEST_ES) as f:
        references = [line.strip() for line in f if line.strip()][:args.n_samples]

    all_results = {}

    for name, path in MODELS.items():
        print(f"\n=== {name} ===")
        model, tokenizer = load_model(path, args.device)

        model_results = []
        # Aggregate stats
        entropies_per_layer = None

        for idx, (src, ref) in enumerate(zip(sources, references)):
            prompt = f"Translate the following English sentence to Spanish: {src}"
            result = logit_lens_single(model, tokenizer, prompt, target_prefix=ref)

            if entropies_per_layer is None:
                n_layers = result["n_layers"]
                entropies_per_layer = np.zeros(n_layers)
                ranks_per_layer = np.zeros(n_layers)
                rank_counts = np.zeros(n_layers)

            for lr in result["layers"]:
                entropies_per_layer[lr["layer"]] += lr["entropy"]
                if lr["target_rank"] is not None:
                    ranks_per_layer[lr["layer"]] += lr["target_rank"]
                    rank_counts[lr["layer"]] += 1

            if idx < 5:  # Save detailed results for first 5 examples
                model_results.append({
                    "source": src[:100],
                    "reference": ref[:100],
                    "layers": result["layers"],
                })

        n_samples = len(sources)
        avg_entropy = (entropies_per_layer / n_samples).tolist()
        avg_rank = np.where(
            rank_counts > 0, ranks_per_layer / rank_counts, -1
        ).tolist()

        all_results[name] = {
            "n_layers": n_layers,
            "n_samples": n_samples,
            "avg_entropy_per_layer": avg_entropy,
            "avg_target_rank_per_layer": avg_rank,
            "sample_details": model_results,
        }

        print(f"  Layers: {n_layers}")
        print(f"  Avg final-layer entropy: {avg_entropy[-1]:.2f}")
        print(f"  Avg final-layer target rank: {avg_rank[-1]:.0f}")

        del model
        torch.cuda.empty_cache()

    with open(RESULTS_DIR / "logit_lens.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    np.savez(
        RESULTS_DIR / "logit_lens.npz",
        **{f"{name}_entropy": np.array(r["avg_entropy_per_layer"])
           for name, r in all_results.items()},
        **{f"{name}_rank": np.array(r["avg_target_rank_per_layer"])
           for name, r in all_results.items()},
    )
    print(f"\nResults saved to {RESULTS_DIR}/logit_lens.*")


if __name__ == "__main__":
    main()
