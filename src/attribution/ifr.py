"""Information Flow Routes (IFR) attribution for layer importance scoring.

Implements the core IFR algorithm from Ferrando & Voita (EMNLP 2024) using
HuggingFace hooks directly, without TransformerLens dependency.

The key idea: at each layer, the residual stream is updated by adding
attention output and FFN output. We measure how much each component
contributes to the resulting vector using proximity-based L1 scoring.

For layer l:
  x^{l,A} = x^{l-1} + Attn_l(x^{l-1})    (residual + attention)
  x^l      = x^{l,A} + FFN_l(x^{l,A})      (post-attn residual + FFN)

Contribution of component y to sum s = y + rest:
  proximity(y, s) = max(-||s - y||_1 + ||s||_1, 0)
  importance(y, s) = proximity(y, s) / sum_k proximity(y_k, s)

Aggregating attention + FFN importance per layer gives a layer importance score.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import BASE_MODEL, NUM_LAYERS
from src.utils import get_device, load_env


def proximity(component: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    """Compute proximity-based contribution score.

    Args:
        component: The additive component [batch, seq, hidden] or [seq, hidden]
        total: The resulting sum vector (same shape)

    Returns:
        Scalar proximity score (averaged over positions and batch).
    """
    # max(-||total - component||_1 + ||total||_1, 0)
    residual_norm = torch.norm(total - component, p=1, dim=-1)
    total_norm = torch.norm(total, p=1, dim=-1)
    prox = torch.clamp(-residual_norm + total_norm, min=0.0)
    return prox.mean()


class IFRScorer:
    """Compute per-layer importance scores using IFR attribution.

    Hooks into the model to capture residual stream states at each layer,
    then computes how much each layer's attention and FFN blocks contribute
    to the residual stream progression.
    """

    def __init__(
        self,
        model_name: str = BASE_MODEL,
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device or str(get_device())
        self.dtype = dtype
        self.model_name = model_name

        load_env()
        print(f"Loading model {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device,
        )
        self.model.eval()

        self.num_layers = self.model.config.num_hidden_layers
        self._activations: dict[str, torch.Tensor] = {}
        self._hooks = []

    def _register_hooks(self):
        """Register forward hooks to capture residual stream states.

        For Cohere/Aya architecture, the layer structure is:
          layer.self_attn -> attention output (before residual add)
          layer.mlp -> FFN output (before residual add)

        We capture:
          - Input to each layer (residual stream before layer)
          - Output of self_attn (attention contribution)
          - Output of mlp (FFN contribution)
          - Output of each layer (residual stream after layer)
        """
        self._clear_hooks()
        self._activations = {}

        layers = self.model.model.layers

        for i, layer in enumerate(layers):
            # Capture layer input (residual stream entering this layer)
            def make_layer_input_hook(layer_idx):
                def hook(module, args, kwargs):
                    # The first positional argument is hidden_states
                    hidden = args[0] if args else kwargs.get("hidden_states")
                    if hidden is not None:
                        self._activations[f"layer_{layer_idx}_input"] = hidden.detach()
                return hook

            h = layer.register_forward_pre_hook(make_layer_input_hook(i), with_kwargs=True)
            self._hooks.append(h)

            # Capture attention output
            def make_attn_hook(layer_idx):
                def hook(module, input, output):
                    # self_attn returns (attn_output, attn_weights, past_kv)
                    attn_out = output[0] if isinstance(output, tuple) else output
                    self._activations[f"layer_{layer_idx}_attn_out"] = attn_out.detach()
                return hook

            h = layer.self_attn.register_forward_hook(make_attn_hook(i))
            self._hooks.append(h)

            # Capture MLP output
            def make_mlp_hook(layer_idx):
                def hook(module, input, output):
                    self._activations[f"layer_{layer_idx}_mlp_out"] = output.detach()
                return hook

            h = layer.mlp.register_forward_hook(make_mlp_hook(i))
            self._hooks.append(h)

            # Capture layer output (residual stream leaving this layer)
            def make_layer_output_hook(layer_idx):
                def hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    self._activations[f"layer_{layer_idx}_output"] = hidden.detach()
                return hook

            h = layer.register_forward_hook(make_layer_output_hook(i))
            self._hooks.append(h)

    def _clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._activations = {}

    @torch.no_grad()
    def score_single(self, text: str) -> dict[str, torch.Tensor]:
        """Compute per-layer importance for a single input.

        Returns:
            Dict with keys:
                - 'attn_importance': [num_layers] attention contribution per layer
                - 'mlp_importance': [num_layers] FFN contribution per layer
                - 'layer_importance': [num_layers] combined (attn + mlp) importance
        """
        self._register_hooks()

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        self.model(**inputs)

        num_layers = self.num_layers
        attn_scores = torch.zeros(num_layers)
        mlp_scores = torch.zeros(num_layers)

        for i in range(num_layers):
            layer_input = self._activations.get(f"layer_{i}_input")
            attn_out = self._activations.get(f"layer_{i}_attn_out")
            mlp_out = self._activations.get(f"layer_{i}_mlp_out")
            layer_output = self._activations.get(f"layer_{i}_output")

            if any(x is None for x in [layer_input, attn_out, mlp_out, layer_output]):
                continue

            # Cast to float32 for numerical stability
            layer_input = layer_input.float()
            attn_out = attn_out.float()
            mlp_out = mlp_out.float()
            layer_output = layer_output.float()

            # Post-attention residual: x^{l,A} = x^{l-1} + attn_out
            post_attn = layer_input + attn_out

            # Contribution of attention to post-attention state
            attn_prox = proximity(attn_out, post_attn)
            residual_prox = proximity(layer_input, post_attn)
            total_prox = attn_prox + residual_prox + 1e-10
            attn_scores[i] = (attn_prox / total_prox).cpu()

            # Contribution of MLP to final layer output: x^l = x^{l,A} + mlp_out
            mlp_prox = proximity(mlp_out, layer_output)
            residual_prox2 = proximity(post_attn, layer_output)
            total_prox2 = mlp_prox + residual_prox2 + 1e-10
            mlp_scores[i] = (mlp_prox / total_prox2).cpu()

        self._clear_hooks()

        layer_scores = attn_scores + mlp_scores

        return {
            "attn_importance": attn_scores,
            "mlp_importance": mlp_scores,
            "layer_importance": layer_scores,
        }

    @torch.no_grad()
    def score_dataset(
        self,
        texts: list[str],
        max_length: int = 256,
    ) -> dict[str, torch.Tensor]:
        """Compute averaged per-layer importance across multiple inputs.

        Args:
            texts: List of input texts (typically formatted as translation prompts).
            max_length: Maximum token length (longer inputs are truncated).

        Returns:
            Averaged importance scores across all inputs.
        """
        all_attn = []
        all_mlp = []
        all_layer = []

        for text in tqdm(texts, desc="IFR scoring"):
            # Truncate if needed
            tokens = self.tokenizer(text, truncation=True, max_length=max_length)
            truncated = self.tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)

            scores = self.score_single(truncated)
            all_attn.append(scores["attn_importance"])
            all_mlp.append(scores["mlp_importance"])
            all_layer.append(scores["layer_importance"])

        return {
            "attn_importance": torch.stack(all_attn).mean(dim=0),
            "mlp_importance": torch.stack(all_mlp).mean(dim=0),
            "layer_importance": torch.stack(all_layer).mean(dim=0),
        }

    def rank_layers(
        self,
        scores: dict[str, torch.Tensor],
        key: str = "layer_importance",
    ) -> list[int]:
        """Return layer indices sorted by importance (least important first).

        The first N entries in the returned list are candidates for pruning.
        """
        importance = scores[key]
        # Sort ascending: least important first
        return torch.argsort(importance).tolist()
