"""Heuristic iterative pruning following Moslem et al.

At each step:
1. For each remaining layer, temporarily remove it.
2. Evaluate chrF++ on a small validation set.
3. Permanently remove the layer whose absence causes the smallest chrF++ drop.
4. Repeat until the target number of layers is reached.
"""

import copy
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import BASE_MODEL, RESULTS_DIR, SRC_LANG_NAME, TGT_LANG_NAME, TRANSLATION_PROMPT
from src.evaluation.metrics import compute_chrf
from src.evaluation.translate import translate_batch
from src.pruning.remove_layers import remove_layers
from src.utils import load_env


def evaluate_without_layer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer_idx: int,
    sources: list[str],
    references: list[str],
    batch_size: int = 8,
    max_new_tokens: int = 256,
) -> float:
    """Temporarily remove a layer and evaluate chrF++.

    Creates a shallow copy of the layer list, removes the target layer,
    evaluates, then restores the original. This avoids full model copies.
    """
    # Save references to original layers instead of deepcopy — the full model
    # is multi-GB, so copying it for each trial would be prohibitively expensive.
    original_layers = list(model.model.layers)
    original_num = model.config.num_hidden_layers

    # Swap in a new ModuleList minus the target layer. The layer objects
    # themselves are not copied; only the container changes.
    new_layers = [l for i, l in enumerate(original_layers) if i != layer_idx]
    model.model.layers = torch.nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)

    # Must re-index for the temporary config so KV cache allocation matches.
    for new_idx, layer in enumerate(new_layers):
        layer.self_attn.layer_idx = new_idx

    try:
        prompts = [
            TRANSLATION_PROMPT.format(
                src_lang=SRC_LANG_NAME,
                tgt_lang=TGT_LANG_NAME,
                source=src,
            )
            for src in sources
        ]
        hypotheses = translate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
        score = compute_chrf(hypotheses, references)
    finally:
        # Restore original layers and reset layer_idx values to their original
        # positions. Without this, the re-indexed values from the trial would
        # corrupt subsequent trials and the final model state.
        model.model.layers = torch.nn.ModuleList(original_layers)
        model.config.num_hidden_layers = original_num
        for orig_idx, layer in enumerate(original_layers):
            layer.self_attn.layer_idx = orig_idx

    return score


def iterative_prune(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sources: list[str],
    references: list[str],
    target_layers: int,
    batch_size: int = 8,
    log_path: Path | None = None,
) -> list[int]:
    """Iteratively remove layers that cause the least chrF++ degradation.

    Args:
        model: The model to prune (modified in-place).
        tokenizer: Model tokenizer.
        sources: Validation source sentences.
        references: Validation reference translations.
        target_layers: Desired final number of layers.
        batch_size: Batch size for translation.
        log_path: Optional path to save pruning log.

    Returns:
        List of removed layer indices (in order of removal).
    """
    removed = []
    log = []

    # Greedy search: at each step we try removing every remaining layer (O(n)
    # evaluations) and keep the one that hurts least. Over k removal steps
    # this is O(n*k) evaluations — expensive but produces good orderings.
    while model.config.num_hidden_layers > target_layers:
        n_layers = model.config.num_hidden_layers
        print(f"\n--- Pruning step: {n_layers} -> {n_layers - 1} layers ---")

        best_layer = -1
        best_score = -1.0

        for layer_idx in range(n_layers):
            score = evaluate_without_layer(
                model, tokenizer, layer_idx, sources, references, batch_size
            )
            print(f"  Layer {layer_idx}: chrF++ = {score:.2f}")

            if score > best_score:
                best_score = score
                best_layer = layer_idx

        print(f"  -> Removing layer {best_layer} (chrF++ = {best_score:.2f})")
        removed.append(best_layer)
        log.append({
            "step": len(removed),
            "layers_before": n_layers,
            "removed_layer": best_layer,
            "chrf_score": best_score,
        })

        # Actually remove the layer
        remove_layers(model, [best_layer])

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump({"removed_layers": removed, "steps": log}, f, indent=2)
        print(f"Pruning log saved to {log_path}")

    return removed
