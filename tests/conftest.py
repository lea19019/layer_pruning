"""Shared fixtures for the test suite.

Provides mock models, tokenizers, temporary directories, and sample data
so that all tests can run on CPU without downloading anything.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Mock transformer layer that mimics Cohere/Aya architecture
# ---------------------------------------------------------------------------


class MockSelfAttn(nn.Module):
    """Minimal self-attention stand-in with a layer_idx attribute."""

    def __init__(self, hidden_size: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, **kwargs):
        attn_out = self.linear(hidden_states)
        return (attn_out,)  # tuple like real attention


class MockMLP(nn.Module):
    """Minimal MLP stand-in."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.linear(hidden_states)


class MockTransformerLayer(nn.Module):
    """Single transformer layer with residual connections."""

    def __init__(self, hidden_size: int, layer_idx: int):
        super().__init__()
        self.self_attn = MockSelfAttn(hidden_size, layer_idx)
        self.mlp = MockMLP(hidden_size)

    def forward(self, hidden_states, **kwargs):
        attn_out = self.self_attn(hidden_states)[0]
        post_attn = hidden_states + attn_out
        mlp_out = self.mlp(post_attn)
        output = post_attn + mlp_out
        return (output,)


class MockInnerModel(nn.Module):
    """Mimics model.model with an nn.ModuleList of layers."""

    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [MockTransformerLayer(hidden_size, i) for i in range(num_layers)]
        )
        self.embed = nn.Embedding(100, hidden_size)

    def forward(self, input_ids, **kwargs):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)[0]
        return h


class MockConfig:
    """Mimics model.config."""

    def __init__(self, num_hidden_layers: int):
        self.num_hidden_layers = num_hidden_layers


class MockCausalLM(nn.Module):
    """Top-level mock that mirrors AutoModelForCausalLM structure."""

    def __init__(self, num_layers: int = 4, hidden_size: int = 16):
        super().__init__()
        self.config = MockConfig(num_layers)
        self.model = MockInnerModel(num_layers, hidden_size)
        self.lm_head = nn.Linear(hidden_size, 100, bias=False)
        self._hidden_size = hidden_size

    def forward(self, input_ids, **kwargs):
        h = self.model(input_ids)
        logits = self.lm_head(h)
        return MagicMock(logits=logits)

    @property
    def device(self):
        return next(self.parameters()).device


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model():
    """Return a small 4-layer mock CausalLM on CPU in float32."""
    model = MockCausalLM(num_layers=4, hidden_size=16)
    model.eval()
    return model


@pytest.fixture
def mock_model_2layer():
    """Return a tiny 2-layer mock CausalLM for IFR tests."""
    model = MockCausalLM(num_layers=2, hidden_size=8)
    model.eval()
    return model


@pytest.fixture
def mock_tokenizer():
    """Return a mock tokenizer that produces simple integer tensors."""
    tok = MagicMock()
    tok.pad_token = "<pad>"
    tok.eos_token = "</s>"
    tok.pad_token_id = 0
    tok.eos_token_id = 1

    class TokenizerOutput(dict):
        """Dict subclass with .to() method to mimic HF BatchEncoding."""
        def to(self, device):
            return TokenizerOutput({
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in self.items()
            })

    def encode_side_effect(text, **kwargs):
        # Return a dict-like object with input_ids
        ids = list(range(1, min(len(text.split()) + 1, 20)))
        result = {"input_ids": ids}
        if kwargs.get("return_tensors") == "pt":
            result = TokenizerOutput({k: torch.tensor([v]) for k, v in result.items()})
            result["attention_mask"] = torch.ones_like(result["input_ids"])
        return result

    tok.side_effect = encode_side_effect
    tok.__call__ = encode_side_effect
    tok.decode = lambda ids, **kw: "mock decoded text"
    tok.encode = lambda text, **kw: list(range(1, min(len(text.split()) + 1, 20)))
    return tok


@pytest.fixture
def sample_pairs():
    """Return a small list of (source, target) pairs."""
    return [
        ("Ahoj svete", "Hallo Welt"),
        ("Dobry den", "Guten Tag"),
        ("Jak se mas", "Wie geht es dir"),
        ("Dekuji", "Danke"),
        ("Prosim", "Bitte"),
    ]


@pytest.fixture
def sample_tsv(tmp_path, sample_pairs):
    """Write sample pairs to a TSV file and return its path."""
    tsv_path = tmp_path / "test.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        for src, tgt in sample_pairs:
            f.write(f"{src}\t{tgt}\n")
    return tsv_path


@pytest.fixture
def sample_parallel_files(tmp_path, sample_pairs):
    """Write sample pairs to parallel .cs and .de files; return (src_path, tgt_path)."""
    src_path = tmp_path / "test.cs"
    tgt_path = tmp_path / "test.de"
    with open(src_path, "w", encoding="utf-8") as f:
        for src, _ in sample_pairs:
            f.write(src + "\n")
    with open(tgt_path, "w", encoding="utf-8") as f:
        for _, tgt in sample_pairs:
            f.write(tgt + "\n")
    return src_path, tgt_path


@pytest.fixture
def sample_ifr_scores(tmp_path):
    """Write a sample IFR scores JSON and return its path."""
    scores = {
        "model": "test-model",
        "n_samples": 5,
        "layer_importance": [0.8, 0.1, 0.3, 0.9],
        "attn_importance": [0.4, 0.05, 0.15, 0.45],
        "mlp_importance": [0.4, 0.05, 0.15, 0.45],
        "ranking_least_important_first": [1, 2, 0, 3],
    }
    path = tmp_path / "ifr_scores.json"
    with open(path, "w") as f:
        json.dump(scores, f)
    return path


@pytest.fixture
def sample_yaml_config(tmp_path):
    """Write a minimal experiment YAML config and return its path."""
    import yaml

    cfg = {
        "experiment_id": "T1",
        "description": "Test experiment",
        "base_model": "test-model",
        "seed": 42,
        "pruning": {"method": "none"},
        "finetuning": {"enabled": False},
        "distillation": {"enabled": False},
        "quantization": {"enabled": False},
    }
    path = tmp_path / "test_config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path
