#!/usr/bin/env python3
"""Rerun M-series en-es experiments with fixed heuristic pruning.

Prunes once from 32 → 16 layers, saving checkpoints at 24, 20, 16.
Then fine-tunes M1 (no KD) and M2 (with KD) from each checkpoint.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TRANSLATION_PROMPT
from src.pruning.heuristic import iterative_prune
from src.pruning.remove_layers import save_pruned_model
from src.run_experiment import run_experiment


def main():
    data_dir = Path("data/filtered_en_es")
    results_base = Path("experiments/results")
    val_size = 200

    # Load validation data
    with open(data_dir / "test.en") as f:
        val_src = f.read().splitlines()[:val_size]
    with open(data_dir / "test.es") as f:
        val_ref = f.read().splitlines()[:val_size]

    # Load base model once
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        "CohereForAI/aya-expanse-8b", torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-expanse-8b")

    # Prune iteratively from 32 → 16, saving at 24, 20, 16
    save_points = {24: "8", 20: "12", 16: "16"}  # layers → n_removed
    pruned_dirs = {}

    print("\n=== Heuristic pruning 32 → 16 (English → Spanish) ===")
    removed_all = []
    remaining_original_ids = list(range(model.config.num_hidden_layers))

    current_layers = model.config.num_hidden_layers  # 32

    while current_layers > 16:
        # Prune one layer
        n_layers = model.config.num_hidden_layers
        print(f"\n--- Pruning step: {n_layers} -> {n_layers - 1} layers ---")

        best_local_idx = -1
        best_score = -1.0

        for local_idx in range(n_layers):
            orig_id = remaining_original_ids[local_idx]
            if orig_id in {0, 31}:
                print(f"  Layer {local_idx} (orig {orig_id}): PROTECTED, skipping")
                continue

            from src.pruning.heuristic import evaluate_without_layer
            score = evaluate_without_layer(
                model, tokenizer, local_idx, val_src, val_ref, batch_size=8,
                src_lang="English", tgt_lang="Spanish",
            )
            print(f"  Layer {local_idx} (orig {orig_id}): chrF++ = {score:.2f}")

            if score > best_score:
                best_score = score
                best_local_idx = local_idx

        best_orig_id = remaining_original_ids[best_local_idx]
        print(f"  -> Removing layer {best_local_idx} (orig {best_orig_id}, chrF++ = {best_score:.2f})")
        removed_all.append(best_orig_id)
        remaining_original_ids.pop(best_local_idx)

        # Actually remove the layer
        from src.pruning.remove_layers import remove_layers
        remove_layers(model, [best_local_idx])
        current_layers = model.config.num_hidden_layers

        # Save checkpoint if at a save point
        if current_layers in save_points:
            n_removed = save_points[current_layers]
            for prefix in ["M1", "M2"]:
                exp_id = f"{prefix}_{n_removed}_enes"
                pruned_dir = results_base / exp_id / "pruned_model"
                pruned_dir.mkdir(parents=True, exist_ok=True)
                save_pruned_model(model, tokenizer, str(pruned_dir))
                pruned_dirs[exp_id] = pruned_dir
                print(f"  Saved {current_layers}-layer model for {exp_id}")

            # Save pruning log
            log = {"removed_layers": removed_all[:], "remaining": remaining_original_ids[:]}
            for prefix in ["M1", "M2"]:
                exp_id = f"{prefix}_{n_removed}_enes"
                with open(results_base / exp_id / "pruning_log.json", "w") as f:
                    json.dump(log, f, indent=2)

    # Free GPU memory from pruning
    del model
    torch.cuda.empty_cache()

    # Now run FT + eval for each experiment using the saved pruned models
    configs = [
        "M1_8_enes", "M1_12_enes", "M1_16_enes",
        "M2_8_enes", "M2_12_enes", "M2_16_enes",
    ]

    for exp_id in configs:
        results_file = results_base / exp_id / "results.json"
        if results_file.exists():
            print(f"\n=== SKIP {exp_id}: already done ===")
            continue

        print(f"\n============================================================")
        print(f"=== Running {exp_id} at FT+eval stage ===")
        print(f"============================================================")

        try:
            run_experiment(f"experiments/configs/{exp_id}.yaml")
        except Exception as e:
            print(f"=== FAILED {exp_id}: {e} ===")

    print("\n=== All done ===")


if __name__ == "__main__":
    main()
