"""CKA (Centered Kernel Alignment) computation utilities.

Adapted from ~/similarity/circuits/geometry.py for use with HuggingFace
models (no TransformerLens dependency).
"""

import numpy as np
import torch


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered Kernel Alignment with linear kernel.

    Args:
        X: (n_examples, d_model_x) activation matrix
        Y: (n_examples, d_model_y) activation matrix

    Returns:
        CKA similarity in [0, 1].
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


def collect_residual_states(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> list[np.ndarray]:
    """Extract residual stream (layer outputs) for each layer via hooks.

    Args:
        model: HuggingFace CausalLM (e.g., Aya Expanse).
        input_ids: (batch, seq_len) token IDs.
        attention_mask: (batch, seq_len) attention mask, optional.

    Returns:
        List of (batch, d_model) numpy arrays, one per layer.
        We take the last non-padding token's representation.
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
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        model(**kwargs)

    for h in hooks:
        h.remove()

    # Extract last-token representation per example
    n_layers = len(layers)
    results = []
    for i in range(n_layers):
        act = activations[i]  # (batch, seq, hidden)
        if attention_mask is not None:
            # Find last non-padding position per example
            seq_lens = attention_mask.sum(dim=1) - 1  # (batch,)
            last_tok = act[torch.arange(act.size(0)), seq_lens]
        else:
            last_tok = act[:, -1, :]
        results.append(last_tok.cpu().float().numpy())

    return results


def collect_all_residuals(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    batch_size: int = 8,
    max_length: int = 256,
) -> np.ndarray:
    """Collect residual stream states for a list of texts.

    Args:
        model: HuggingFace CausalLM.
        tokenizer: Associated tokenizer.
        texts: Input texts.
        batch_size: Batch size for processing.
        max_length: Maximum sequence length.

    Returns:
        (n_layers, n_examples, d_model) numpy array.
    """
    all_layer_acts = None
    device = next(model.parameters()).device

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        layer_acts = collect_residual_states(
            model, inputs["input_ids"], inputs["attention_mask"]
        )

        if all_layer_acts is None:
            n_layers = len(layer_acts)
            all_layer_acts = [[] for _ in range(n_layers)]

        for i, act in enumerate(layer_acts):
            all_layer_acts[i].append(act)

    # Concatenate batches: (n_layers, n_examples, d_model)
    return np.array([np.concatenate(acts, axis=0) for acts in all_layer_acts])


def pairwise_cka_matrix(activations: np.ndarray) -> np.ndarray:
    """Compute pairwise CKA between all layers of a single model.

    Args:
        activations: (n_layers, n_examples, d_model)

    Returns:
        (n_layers, n_layers) CKA matrix.
    """
    n_layers = activations.shape[0]
    cka_matrix = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(i, n_layers):
            score = linear_cka(activations[i], activations[j])
            cka_matrix[i, j] = score
            cka_matrix[j, i] = score
    return cka_matrix


def cross_model_cka(
    acts_a: np.ndarray, acts_b: np.ndarray
) -> np.ndarray:
    """Compute CKA between all layer pairs of two models.

    Args:
        acts_a: (n_layers_a, n_examples, d_model_a)
        acts_b: (n_layers_b, n_examples, d_model_b)

    Returns:
        (n_layers_a, n_layers_b) CKA matrix.
    """
    n_a, n_b = acts_a.shape[0], acts_b.shape[0]
    n_examples = min(acts_a.shape[1], acts_b.shape[1])
    cka_matrix = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            cka_matrix[i, j] = linear_cka(
                acts_a[i, :n_examples], acts_b[j, :n_examples]
            )
    return cka_matrix
