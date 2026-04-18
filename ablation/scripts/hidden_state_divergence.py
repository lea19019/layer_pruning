"""Experiment 1: Hidden state divergence analysis via CKA.

Compares residual stream representations across all 4 ablation study models
at matched relative depths. Answers: where in the network do pruned and
unpruned models diverge, and does fine-tuning bring them back?

Models compared:
  - Base (unpruned, no FT): raw Aya 8B
  - Pruned only (IP_16_enes): 16 layers, no FT
  - Pruned + FT + KD (I2_16_enes): 16 layers, fine-tuned
  - Unpruned + FT + KD (B4_enes): 32 layers, fine-tuned (ceiling)

Usage:
    python -m ablation.scripts.hidden_state_divergence [--n-samples 100] [--batch-size 4]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ablation.scripts.cka import collect_all_residuals, cross_model_cka, linear_cka

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODELS = {
    "base": "CohereForAI/aya-expanse-8b",
    "pruned_only": str(PROJECT_ROOT / "experiments/results/IP_16_enes/pruned_model"),
    "pruned_ft_kd": str(
        PROJECT_ROOT / "experiments/results/I2_16_enes/finetuned/merged"
    ),
    "full_ft_kd": str(PROJECT_ROOT / "experiments/results/B4_enes/finetuned/merged"),
}

TEST_EN = PROJECT_ROOT / "data/filtered_en_es/test.en"
RESULTS_DIR = PROJECT_ROOT / "ablation/results"


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

    # Collect activations for each model
    all_acts = {}
    for name, path in MODELS.items():
        print(f"\n--- Loading {name}: {path} ---")
        model, tokenizer = load_model(path, args.device)
        acts = collect_all_residuals(
            model, tokenizer, prompts, batch_size=args.batch_size
        )
        all_acts[name] = acts
        n_layers = acts.shape[0]
        print(f"  {name}: {n_layers} layers, shape {acts.shape}")
        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Save raw activations
    np.savez_compressed(
        RESULTS_DIR / "activations_all_models.npz",
        **{f"{name}_acts": acts for name, acts in all_acts.items()},
    )
    print("\nSaved activations")

    # Compute cross-model CKA matrices
    results = {}
    model_names = list(all_acts.keys())
    for i, name_a in enumerate(model_names):
        for j, name_b in enumerate(model_names):
            if j <= i:
                continue
            print(f"\nComputing CKA: {name_a} vs {name_b}")
            cka = cross_model_cka(all_acts[name_a], all_acts[name_b])
            key = f"{name_a}_vs_{name_b}"
            results[key] = cka
            np.save(RESULTS_DIR / f"cka_{key}.npy", cka)
            print(f"  Shape: {cka.shape}, diagonal mean: {np.diag(cka).mean():.3f}")

    # Compute matched-depth CKA (diagonal) for 16-layer models vs 32-layer models
    # Map 16 layers to relative depths [0, 1], find closest 32-layer match
    for pruned_name in ["pruned_only", "pruned_ft_kd"]:
        for full_name in ["base", "full_ft_kd"]:
            n_pruned = all_acts[pruned_name].shape[0]
            n_full = all_acts[full_name].shape[0]
            matched_cka = []
            for pi in range(n_pruned):
                rel_depth = pi / (n_pruned - 1)
                fi = round(rel_depth * (n_full - 1))
                score = linear_cka(
                    all_acts[pruned_name][pi], all_acts[full_name][fi]
                )
                matched_cka.append(
                    {"pruned_layer": pi, "full_layer": fi, "rel_depth": rel_depth, "cka": score}
                )
            key = f"matched_{pruned_name}_vs_{full_name}"
            results[key] = matched_cka
            print(f"\nMatched-depth CKA ({pruned_name} vs {full_name}):")
            for m in matched_cka:
                print(f"  Layer {m['pruned_layer']}/{m['full_layer']} "
                      f"(depth {m['rel_depth']:.2f}): CKA={m['cka']:.3f}")

    # Save summary
    summary = {}
    for key, val in results.items():
        if isinstance(val, np.ndarray):
            summary[key] = {"shape": list(val.shape), "mean": float(val.mean())}
        elif isinstance(val, list):
            summary[key] = val
    with open(RESULTS_DIR / "hidden_state_divergence.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. Results saved to ablation/results/")


if __name__ == "__main__":
    main()
