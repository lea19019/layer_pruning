"""Layer removal mechanics for Cohere/Aya architecture.

Given a list of layer indices to remove, this module physically removes those
layers from the model and updates the config. The pruned model can then be
saved and loaded as a standard HuggingFace model.
"""

import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from src.utils import load_env


def remove_layers(
    model: PreTrainedModel,
    layers_to_remove: list[int],
) -> PreTrainedModel:
    """Remove specified layers from the model in-place.

    Args:
        model: HuggingFace CausalLM model.
        layers_to_remove: Sorted list of layer indices to remove.

    Returns:
        The modified model with layers removed and config updated.
    """
    # Deduplicate and sort in reverse order so that removing a high-index
    # layer doesn't shift lower-index layers and invalidate remaining indices.
    layers_to_remove = sorted(set(layers_to_remove), reverse=True)
    layer_list = model.model.layers

    original_count = len(layer_list)
    # Validate all indices against the original layer count before any deletion.
    for idx in layers_to_remove:
        if idx < 0 or idx >= original_count:
            raise ValueError(f"Layer index {idx} out of range [0, {original_count})")
    for idx in layers_to_remove:
        del layer_list[idx]

    model.config.num_hidden_layers = len(layer_list)

    # Re-index remaining layers so self_attn.layer_idx matches position.
    # This is critical: HF's KV cache is allocated as a list of length
    # num_hidden_layers, indexed by layer_idx. A stale layer_idx (e.g. 39
    # in a now-35-layer model) causes an index-out-of-range crash during
    # generation.
    for new_idx, layer in enumerate(layer_list):
        layer.self_attn.layer_idx = new_idx

    new_count = len(layer_list)
    print(f"Removed {original_count - new_count} layers: {original_count} -> {new_count}")
    return model


def load_and_prune(
    model_name: str,
    layers_to_remove: list[int],
    device_map: str = "auto",
    dtype: torch.dtype = torch.float16,
) -> tuple[PreTrainedModel, AutoTokenizer]:
    """Load a model and remove specified layers.

    Args:
        model_name: HuggingFace model name or path.
        layers_to_remove: List of layer indices to remove.
        device_map: Device placement strategy.
        dtype: Model dtype.

    Returns:
        Tuple of (pruned_model, tokenizer).
    """
    load_env()

    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
    )

    model = remove_layers(model, layers_to_remove)
    return model, tokenizer


def save_pruned_model(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    output_dir: str,
):
    """Save the pruned model and tokenizer."""
    print(f"Saving pruned model to {output_dir} ...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")
