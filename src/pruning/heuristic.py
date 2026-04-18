"""Heuristic iterative pruning replicating Moslem et al.

Replicates the approach from:
  Moslem et al., "Iterative Layer Pruning for Efficient Translation Inference"
  (WMT 2025). Code: github.com/ymoslem/Model-Compression

At each step:
1. For each remaining layer (excluding first and last), temporarily remove it.
2. Evaluate chrF++ on a validation set using chat-templated prompts.
3. Permanently remove the layer whose absence causes the smallest chrF++ drop.
4. Repeat until the target number of layers is reached.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import SRC_LANG_NAME, TGT_LANG_NAME, TRANSLATION_PROMPT
from src.evaluation.metrics import compute_chrf
from src.evaluation.translate import translate_batch_chat
from src.pruning.remove_layers import remove_layers


# Moslem skips the first and last layers (they are never pruning candidates).
PROTECTED_LAYERS = {0, 31}


def evaluate_without_layer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer_idx: int,
    sources: list[str],
    references: list[str],
    batch_size: int = 8,
    max_new_tokens: int = 256,
    src_lang: str = SRC_LANG_NAME,
    tgt_lang: str = TGT_LANG_NAME,
) -> float:
    """Temporarily remove a layer and evaluate chrF++.

    Creates a shallow copy of the layer list, removes the target layer,
    evaluates, then restores the original. This avoids full model copies.
    """
    original_layers = list(model.model.layers)
    original_num = model.config.num_hidden_layers

    # Swap in a new ModuleList minus the target layer.
    new_layers = [l for i, l in enumerate(original_layers) if i != layer_idx]
    model.model.layers = torch.nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)

    # Must re-index for the temporary config so KV cache allocation matches.
    for new_idx, layer in enumerate(new_layers):
        layer.self_attn.layer_idx = new_idx

    try:
        prompts = [
            TRANSLATION_PROMPT.format(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                source=src,
            )
            for src in sources
        ]
        hypotheses = translate_batch_chat(
            model, tokenizer, prompts, max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )
        score = compute_chrf(hypotheses, references)
    finally:
        # Restore original layers and reset layer_idx values.
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
    src_lang: str = SRC_LANG_NAME,
    tgt_lang: str = TGT_LANG_NAME,
) -> list[int]:
    """Iteratively remove layers that cause the least chrF++ degradation.

    Following Moslem et al., the first and last layers of the original model
    are never considered for removal.

    Args:
        model: The model to prune (modified in-place).
        tokenizer: Model tokenizer.
        sources: Validation source sentences.
        references: Validation reference translations.
        target_layers: Desired final number of layers.
        batch_size: Batch size for translation.
        log_path: Optional path to save pruning log.

    Returns:
        List of removed layer indices (in original-model numbering).
    """
    removed = []
    log = []

    # Track which original-model layer indices are still present.
    # This lets us report removed layers in original numbering (like Moslem).
    remaining_original_ids = list(range(model.config.num_hidden_layers))

    while model.config.num_hidden_layers > target_layers:
        n_layers = model.config.num_hidden_layers
        print(f"\n--- Pruning step: {n_layers} -> {n_layers - 1} layers ---")

        best_local_idx = -1
        best_score = -1.0

        for local_idx in range(n_layers):
            orig_id = remaining_original_ids[local_idx]
            # Skip protected layers (first and last of the original model).
            if orig_id in PROTECTED_LAYERS:
                print(f"  Layer {local_idx} (orig {orig_id}): PROTECTED, skipping")
                continue

            score = evaluate_without_layer(
                model, tokenizer, local_idx, sources, references, batch_size,
                src_lang=src_lang, tgt_lang=tgt_lang,
            )
            print(f"  Layer {local_idx} (orig {orig_id}): chrF++ = {score:.2f}")

            if score > best_score:
                best_score = score
                best_local_idx = local_idx

        best_orig_id = remaining_original_ids[best_local_idx]
        print(f"  -> Removing layer {best_local_idx} (orig {best_orig_id}, chrF++ = {best_score:.2f})")
        removed.append(best_orig_id)
        remaining_original_ids.pop(best_local_idx)
        log.append({
            "step": len(removed),
            "layers_before": n_layers,
            "removed_layer_original": best_orig_id,
            "removed_layer_local": best_local_idx,
            "chrf_score": best_score,
        })

        # Actually remove the layer
        remove_layers(model, [best_local_idx])

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump({"removed_layers": removed, "steps": log}, f, indent=2)
        print(f"Pruning log saved to {log_path}")

    return removed
