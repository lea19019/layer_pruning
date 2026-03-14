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
    layers_to_remove = sorted(layers_to_remove, reverse=True)
    layer_list = model.model.layers

    original_count = len(layer_list)
    for idx in layers_to_remove:
        if idx < 0 or idx >= len(layer_list):
            raise ValueError(f"Layer index {idx} out of range [0, {len(layer_list)})")
        del layer_list[idx]

    # Update config
    model.config.num_hidden_layers = len(layer_list)

    # Re-index remaining layers so self_attn.layer_idx matches position.
    # Without this, the KV cache (which is sized by num_hidden_layers)
    # gets an out-of-range index from layers that kept their old index.
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
        torch_dtype=dtype,
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
