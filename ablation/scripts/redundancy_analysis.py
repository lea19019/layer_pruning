"""Experiment 10: Redundancy analysis via pairwise CKA within a single model.

Computes pairwise CKA between all 32 layers of the full+FT model (B4_enes)
and optionally the base model. If adjacent middle layers show CKA > 0.95,
this provides theoretical justification for pruning: those layers are
functionally redundant.

Also computes effective rank (via SVD) at each layer to measure information
compression.

Usage:
    python -m ablation.scripts.redundancy_analysis [--n-samples 100] [--batch-size 4]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ablation.scripts.cka import collect_all_residuals, pairwise_cka_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "ablation/results"

MODELS = {
    "base": "CohereForAI/aya-expanse-8b",
    "full_ft_kd": str(PROJECT_ROOT / "experiments/results/B4_enes/finetuned/merged"),
}

TEST_EN = PROJECT_ROOT / "data/filtered_en_es/test.en"


def load_test_sentences(path: Path, n: int) -> list[str]:
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[:n]


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


def effective_rank(activations: np.ndarray) -> np.ndarray:
    """Compute effective rank at each layer via SVD.

    Effective rank = exp(entropy of normalized singular values).
    Higher means the layer uses more dimensions of the representation space.

    Args:
        activations: (n_layers, n_examples, d_model)

    Returns:
        (n_layers,) effective rank per layer.
    """
    n_layers = activations.shape[0]
    ranks = np.zeros(n_layers)
    for i in range(n_layers):
        _, s, _ = np.linalg.svd(activations[i], full_matrices=False)
        # Normalize singular values to form a probability distribution
        s = s / s.sum()
        # Remove zeros to avoid log(0)
        s = s[s > 1e-10]
        # Shannon entropy -> effective rank
        entropy = -np.sum(s * np.log(s))
        ranks[i] = np.exp(entropy)
    return ranks


def format_prompts(sentences: list[str]) -> list[str]:
    return [
        f"Translate the following English sentence to Spanish: {s}" for s in sentences
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sentences = load_test_sentences(TEST_EN, args.n_samples)
    prompts = format_prompts(sentences)
    print(f"Loaded {len(prompts)} prompts")

    results = {}

    for name, path in MODELS.items():
        print(f"\n=== {name}: {path} ===")
        model, tokenizer = load_model(path, args.device)
        acts = collect_all_residuals(
            model, tokenizer, prompts, batch_size=args.batch_size
        )
        n_layers = acts.shape[0]
        print(f"  Layers: {n_layers}, shape: {acts.shape}")

        # Pairwise CKA
        print("  Computing pairwise CKA...")
        cka = pairwise_cka_matrix(acts)
        np.save(RESULTS_DIR / f"pairwise_cka_{name}.npy", cka)

        # Adjacent layer CKA
        adjacent_cka = [float(cka[i, i + 1]) for i in range(n_layers - 1)]
        print(f"  Adjacent CKA: min={min(adjacent_cka):.3f}, "
              f"max={max(adjacent_cka):.3f}, mean={np.mean(adjacent_cka):.3f}")

        # Find highly redundant pairs (CKA > 0.95)
        redundant = []
        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                if cka[i, j] > 0.95:
                    redundant.append(
                        {"layer_a": i, "layer_b": j, "cka": round(float(cka[i, j]), 4)}
                    )
        print(f"  Redundant pairs (CKA > 0.95): {len(redundant)}")

        # Effective rank
        print("  Computing effective rank...")
        erank = effective_rank(acts)
        np.save(RESULTS_DIR / f"effective_rank_{name}.npy", erank)
        print(f"  Effective rank: min={erank.min():.1f}, max={erank.max():.1f}, "
              f"mean={erank.mean():.1f}")

        results[name] = {
            "n_layers": n_layers,
            "adjacent_cka": adjacent_cka,
            "n_redundant_pairs": len(redundant),
            "redundant_pairs": redundant[:20],  # top 20
            "effective_rank": erank.tolist(),
            "cka_mean": float(cka.mean()),
            "cka_diagonal_mean": float(np.diag(cka).mean()),
        }

        del model
        torch.cuda.empty_cache()

    # Save summary
    with open(RESULTS_DIR / "redundancy_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
