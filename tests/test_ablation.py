"""Tests for ablation analysis scripts.

All tests run on CPU with mock models — no GPU or downloads required.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from ablation.scripts.cka import (
    collect_residual_states,
    cross_model_cka,
    linear_cka,
    pairwise_cka_matrix,
)
from ablation.scripts.output_categorization import (
    categorize_translation,
    detect_language_heuristic,
    detect_repetition,
)
from ablation.scripts.redundancy_analysis import effective_rank
from ablation.scripts.logit_lens import logit_lens_single
from ablation.scripts.attention_comparison import (
    attention_entropy,
    attention_concentration,
)
from ablation.scripts.weight_diff_analysis import extract_layer_info, compute_weight_diffs
from ablation.scripts.ft_recovery_curve import build_dataset, load_parallel_data
from ablation.scripts.surgical_fix import compute_rms_scale, fit_linear_probe
from ablation.scripts.surgical_fix_v2 import (
    compute_per_layer_scales,
    orthogonal_procrustes,
    reduced_rank_regression,
)


# ── CKA math tests ─────────────────────────────────────────────────────────


class TestLinearCKA:
    def test_identical_inputs(self):
        """CKA of a matrix with itself should be 1.0."""
        X = np.random.randn(50, 32)
        assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_inputs(self):
        """CKA of orthogonal matrices should be near 0."""
        # Use larger sample size to reduce cross-correlation noise between
        # the "independent" random fills of the two subspaces.
        rng = np.random.RandomState(0)
        X = np.zeros((500, 10))
        Y = np.zeros((500, 10))
        X[:, :5] = rng.randn(500, 5)
        Y[:, 5:] = rng.randn(500, 5)
        score = linear_cka(X, Y)
        assert score < 0.15

    def test_range(self):
        """CKA should be in [0, 1]."""
        X = np.random.randn(30, 16)
        Y = np.random.randn(30, 16)
        score = linear_cka(X, Y)
        assert 0.0 <= score <= 1.0

    def test_symmetry(self):
        """CKA(X, Y) == CKA(Y, X)."""
        X = np.random.randn(40, 20)
        Y = np.random.randn(40, 20)
        assert linear_cka(X, Y) == pytest.approx(linear_cka(Y, X), abs=1e-10)

    def test_different_dimensions(self):
        """CKA should work with different d_model between X and Y."""
        X = np.random.randn(30, 16)
        Y = np.random.randn(30, 32)
        score = linear_cka(X, Y)
        assert 0.0 <= score <= 1.0

    def test_zero_input(self):
        """CKA with all-zero input returns 0."""
        X = np.zeros((10, 5))
        Y = np.random.randn(10, 5)
        assert linear_cka(X, Y) == 0.0

    def test_scaled_inputs(self):
        """CKA should be invariant to scaling."""
        X = np.random.randn(50, 16)
        Y = X * 5.0  # scaled version
        assert linear_cka(X, Y) == pytest.approx(1.0, abs=1e-6)


class TestPairwiseCKA:
    def test_shape(self):
        acts = np.random.randn(4, 20, 16)
        cka = pairwise_cka_matrix(acts)
        assert cka.shape == (4, 4)

    def test_diagonal_is_one(self):
        acts = np.random.randn(4, 30, 16)
        cka = pairwise_cka_matrix(acts)
        for i in range(4):
            assert cka[i, i] == pytest.approx(1.0, abs=1e-5)

    def test_symmetric(self):
        acts = np.random.randn(3, 20, 8)
        cka = pairwise_cka_matrix(acts)
        np.testing.assert_allclose(cka, cka.T, atol=1e-10)


class TestCrossModelCKA:
    def test_shape(self):
        acts_a = np.random.randn(4, 20, 16)
        acts_b = np.random.randn(6, 20, 32)
        cka = cross_model_cka(acts_a, acts_b)
        assert cka.shape == (4, 6)

    def test_same_model(self):
        """Cross-model CKA with same activations should have 1.0 on diagonal."""
        acts = np.random.randn(4, 30, 16)
        cka = cross_model_cka(acts, acts)
        for i in range(4):
            assert cka[i, i] == pytest.approx(1.0, abs=1e-5)


# ── Hook-based activation extraction tests ──────────────────────────────────


class TestCollectResidualStates:
    def test_returns_correct_layers(self, mock_model):
        """Should return one array per layer."""
        input_ids = torch.tensor([[1, 2, 3]])
        acts = collect_residual_states(mock_model, input_ids)
        assert len(acts) == 4  # mock_model has 4 layers

    def test_output_shape(self, mock_model):
        """Each array should be (batch, d_model)."""
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        acts = collect_residual_states(mock_model, input_ids)
        for act in acts:
            assert act.shape == (2, 16)  # batch=2, hidden_size=16

    def test_with_attention_mask(self, mock_model):
        """Should use attention mask to find last non-padding token."""
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        acts = collect_residual_states(mock_model, input_ids, attention_mask)
        assert len(acts) == 4
        for act in acts:
            assert act.shape == (2, 16)

    def test_hooks_cleaned_up(self, mock_model):
        """Hooks should be removed after collection."""
        input_ids = torch.tensor([[1, 2, 3]])
        collect_residual_states(mock_model, input_ids)
        # Run model again — should work without hooks interfering
        output = mock_model(input_ids)
        assert output.logits is not None

    def test_different_inputs_different_activations(self, mock_model):
        """Different inputs should produce different activations."""
        acts1 = collect_residual_states(mock_model, torch.tensor([[1, 2, 3]]))
        acts2 = collect_residual_states(mock_model, torch.tensor([[7, 8, 9]]))
        # At least one layer should differ
        any_different = any(
            not np.allclose(a1, a2, atol=1e-5) for a1, a2 in zip(acts1, acts2)
        )
        assert any_different


# ── Output categorization tests ─────────────────────────────────────────────


class TestDetectLanguage:
    def test_spanish(self):
        text = "El presidente de la nacion dijo que los cambios son para el bien de todos"
        assert detect_language_heuristic(text) == "es"

    def test_english(self):
        text = "The president of the nation said that the changes are for the good of everyone"
        assert detect_language_heuristic(text) == "en"

    def test_short_text(self):
        """Short text may not have enough markers."""
        result = detect_language_heuristic("hola")
        assert result in ("es", "en", "unknown")


class TestDetectRepetition:
    def test_no_repetition(self):
        text = "This is a sentence with no repeated phrases at all in the text."
        score = detect_repetition(text)
        assert score < 0.3

    def test_high_repetition(self):
        text = "the cat sat the cat sat the cat sat the cat sat the cat sat"
        score = detect_repetition(text)
        assert score > 0.5

    def test_short_text(self):
        assert detect_repetition("hi") == 0.0


class TestCategorizeTranslation:
    def test_good_spanish_translation(self):
        result = categorize_translation(
            source="The cat is on the table",
            hypothesis="El gato esta en la mesa de la cocina",
            reference="El gato esta sobre la mesa",
        )
        assert "plausible" in result["categories"]

    def test_wrong_language(self):
        result = categorize_translation(
            source="The cat is on the table",
            hypothesis="The cat is sitting on the table in the room with the door",
            reference="El gato esta sobre la mesa",
        )
        assert "wrong_language" in result["categories"]

    def test_garbage(self):
        result = categorize_translation(
            source="The cat is on the table",
            hypothesis="",
            reference="El gato esta sobre la mesa",
        )
        assert "garbage" in result["categories"]

    def test_truncation(self):
        result = categorize_translation(
            source="A very long source sentence about many different topics",
            hypothesis="Un",
            reference="Una oracion fuente muy larga sobre muchos temas diferentes que habla de cosas",
        )
        # Either garbage (too short) or truncation
        cats = result["categories"]
        assert "garbage" in cats or "truncation" in cats

    def test_repetition(self):
        repeated = "el gato el gato el gato el gato el gato el gato el gato el gato"
        result = categorize_translation(
            source="The cat",
            hypothesis=repeated,
            reference="El gato",
        )
        assert "repetition" in result["categories"]

    def test_returns_details(self):
        result = categorize_translation(
            source="Hello world",
            hypothesis="Hola mundo de la tierra",
            reference="Hola mundo",
        )
        assert "details" in result
        assert "length_ratio_vs_ref" in result["details"]
        assert "repetition_score" in result["details"]


# ── Effective rank tests ────────────────────────────────────────────────────


class TestEffectiveRank:
    def test_shape(self):
        acts = np.random.randn(4, 30, 16)
        ranks = effective_rank(acts)
        assert ranks.shape == (4,)

    def test_positive(self):
        acts = np.random.randn(3, 20, 8)
        ranks = effective_rank(acts)
        assert all(r > 0 for r in ranks)

    def test_low_rank_data(self):
        """Data with one dominant direction should have low effective rank."""
        acts = np.zeros((1, 50, 16))
        # All examples point in the same direction
        acts[0, :, 0] = np.random.randn(50)
        acts[0, :, 1:] = np.random.randn(50, 15) * 0.01
        ranks = effective_rank(acts)
        assert ranks[0] < 5  # should be near 1

    def test_full_rank_data(self):
        """Random data should have high effective rank."""
        acts = np.random.randn(1, 100, 8)
        ranks = effective_rank(acts)
        assert ranks[0] > 3  # random 8-dim should use most dimensions


# ── Logit lens tests ───────────────────────────────────────────────────────


class TestLogitLens:
    def test_returns_all_layers(self, mock_model, mock_tokenizer):
        """Should return results for each layer."""
        result = logit_lens_single(mock_model, mock_tokenizer, "hello world")
        assert result["n_layers"] == 4
        assert len(result["layers"]) == 4

    def test_layer_fields(self, mock_model, mock_tokenizer):
        """Each layer result should have expected fields."""
        result = logit_lens_single(mock_model, mock_tokenizer, "test prompt")
        for lr in result["layers"]:
            assert "layer" in lr
            assert "rel_depth" in lr
            assert "top5_tokens" in lr
            assert "top5_probs" in lr
            assert "entropy" in lr
            assert len(lr["top5_tokens"]) == 5

    def test_entropy_positive(self, mock_model, mock_tokenizer):
        """Entropy should be non-negative."""
        result = logit_lens_single(mock_model, mock_tokenizer, "some text here")
        for lr in result["layers"]:
            assert lr["entropy"] >= 0

    def test_rel_depth_range(self, mock_model, mock_tokenizer):
        """Relative depth should span [0, 1]."""
        result = logit_lens_single(mock_model, mock_tokenizer, "check depth")
        depths = [lr["rel_depth"] for lr in result["layers"]]
        assert depths[0] == pytest.approx(0.0, abs=0.01)
        assert depths[-1] == pytest.approx(1.0, abs=0.01)


# ── Attention comparison tests ──────────────────────────────────────────────


class TestAttentionEntropy:
    def test_uniform_attention(self):
        """Uniform attention should have maximum entropy."""
        n_heads, seq_len = 4, 10
        # Uniform attention: each position gets 1/seq_len
        attn = np.ones((n_heads, seq_len, seq_len)) / seq_len
        entropy = attention_entropy(attn)
        max_entropy = np.log(seq_len)
        assert entropy == pytest.approx(max_entropy, abs=0.01)

    def test_focused_attention(self):
        """Attention focused on one position should have low entropy."""
        n_heads, seq_len = 2, 8
        attn = np.zeros((n_heads, seq_len, seq_len))
        attn[:, :, 0] = 1.0  # all attend to position 0
        entropy = attention_entropy(attn)
        assert entropy < 0.1  # near-zero entropy

    def test_nonnegative(self):
        """Entropy should always be non-negative."""
        attn = np.random.dirichlet(np.ones(5), size=(3, 5))
        # shape: (3, 5, 5) - 3 heads, 5 positions
        entropy = attention_entropy(attn)
        assert entropy >= 0


class TestAttentionConcentration:
    def test_all_on_bos(self):
        """If all attention is on BOS, bos_frac should be ~1.0."""
        attn = np.zeros((2, 5, 5))
        attn[:, :, 0] = 1.0
        conc = attention_concentration(attn)
        assert conc["bos_frac"] == pytest.approx(1.0, abs=0.01)

    def test_returns_expected_keys(self):
        attn = np.random.dirichlet(np.ones(6), size=(3, 6))
        conc = attention_concentration(attn)
        assert "bos_frac" in conc
        assert "recent_frac" in conc
        assert "diag_frac" in conc

    def test_empty_sequence(self):
        """Should handle empty sequence gracefully."""
        attn = np.zeros((2, 0, 0))
        conc = attention_concentration(attn)
        assert conc["bos_frac"] == 0.0


# ── Weight diff analysis tests ──────────────────────────────────────────────


class TestExtractLayerInfo:
    def test_attn_layer(self):
        idx, mod = extract_layer_info("model.layers.5.self_attn.q_proj.weight")
        assert idx == 5
        assert mod == "attn"

    def test_mlp_layer(self):
        idx, mod = extract_layer_info("model.layers.12.mlp.gate_proj.weight")
        assert idx == 12
        assert mod == "mlp"

    def test_embed(self):
        idx, mod = extract_layer_info("model.embed_tokens.weight")
        assert idx is None
        assert mod == "embed"

    def test_lm_head(self):
        idx, mod = extract_layer_info("lm_head.weight")
        assert idx is None
        assert mod == "head"

    def test_norm(self):
        idx, mod = extract_layer_info("model.layers.3.input_layernorm.weight")
        assert idx == 3
        assert mod == "norm"


class TestComputeWeightDiffs:
    def test_identical_weights(self):
        """Identical weights should have zero diff."""
        w = {"a.weight": torch.randn(4, 4), "b.weight": torch.randn(3, 3)}
        result = compute_weight_diffs(w, w)
        for p in result["per_param"]:
            assert p["diff_norm"] == pytest.approx(0.0, abs=1e-5)

    def test_different_weights(self):
        """Different weights should have nonzero diff."""
        w1 = {"model.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4)}
        w2 = {"model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4)}
        result = compute_weight_diffs(w1, w2)
        assert len(result["per_param"]) == 1
        assert result["per_param"][0]["diff_norm"] > 0

    def test_per_layer_tracking(self):
        """Should group diffs by layer index."""
        w1 = {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4),
            "model.layers.1.mlp.gate_proj.weight": torch.zeros(3, 3),
        }
        w2 = {
            "model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
            "model.layers.1.mlp.gate_proj.weight": torch.ones(3, 3),
        }
        result = compute_weight_diffs(w1, w2)
        assert "0" in result["per_layer"]
        assert "1" in result["per_layer"]


# ── FT recovery curve data loading tests ────────────────────────────────────


class TestFTRecoveryData:
    def test_build_dataset(self):
        """Should create HF dataset with text field."""
        pairs = [("Hello", "Hola"), ("World", "Mundo")]
        ds = build_dataset(pairs)
        assert "text" in ds.column_names
        assert len(ds) == 2
        assert "Translate" in ds[0]["text"]
        assert "Hola" in ds[0]["text"]

    def test_load_parallel_data_fraction(self, tmp_path):
        """Subsampling should reduce data size."""
        src = tmp_path / "src.txt"
        tgt = tmp_path / "tgt.txt"
        src.write_text("\n".join([f"source {i}" for i in range(100)]) + "\n")
        tgt.write_text("\n".join([f"target {i}" for i in range(100)]) + "\n")

        pairs_full = load_parallel_data(src, tgt, fraction=1.0)
        pairs_half = load_parallel_data(src, tgt, fraction=0.5)
        assert len(pairs_full) == 100
        assert len(pairs_half) == 50

    def test_load_parallel_data_minimum(self, tmp_path):
        """Even with tiny fraction, should get at least 1 pair."""
        src = tmp_path / "src.txt"
        tgt = tmp_path / "tgt.txt"
        src.write_text("hello\n")
        tgt.write_text("hola\n")
        pairs = load_parallel_data(src, tgt, fraction=0.01)
        assert len(pairs) >= 1


# ── Surgical fix math tests ─────────────────────────────────────────────────


class TestComputeRmsScale:
    def test_same_residuals_scale_is_one(self):
        """Identical distributions should produce scale=1."""
        r = np.random.randn(100, 16)
        scale = compute_rms_scale(r, r)
        assert scale == pytest.approx(1.0, abs=1e-5)

    def test_scaled_target_recovered(self):
        """If target = 2*pruned, scale should be 2."""
        pruned = np.random.randn(100, 16)
        target = 2.0 * pruned
        scale = compute_rms_scale(pruned, target)
        assert scale == pytest.approx(2.0, abs=1e-4)

    def test_zero_pruned_safe(self):
        """Zero pruned residuals shouldn't NaN."""
        pruned = np.zeros((10, 8))
        target = np.random.randn(10, 8)
        scale = compute_rms_scale(pruned, target)
        assert np.isfinite(scale)


class TestFitLinearProbe:
    def test_identity_if_same(self):
        """If Y=X, T should be close to identity."""
        np.random.seed(0)
        X = np.random.randn(500, 16)
        T = fit_linear_probe(X, X, ridge=1e-6)
        np.testing.assert_allclose(T, np.eye(16), atol=0.01)

    def test_shape(self):
        X = np.random.randn(100, 32)
        Y = np.random.randn(100, 32)
        T = fit_linear_probe(X, Y)
        assert T.shape == (32, 32)

    def test_linear_map_recovered(self):
        """If Y = X @ A, T should approximate A."""
        np.random.seed(0)
        A = np.random.randn(16, 16)
        X = np.random.randn(500, 16)
        Y = X @ A
        T = fit_linear_probe(X, Y, ridge=1e-6)
        np.testing.assert_allclose(T, A, atol=0.05)

    def test_ridge_regularization(self):
        """Higher ridge should shrink T."""
        np.random.seed(0)
        X = np.random.randn(50, 16)
        Y = np.random.randn(50, 16)
        T_low = fit_linear_probe(X, Y, ridge=1e-6)
        T_high = fit_linear_probe(X, Y, ridge=100.0)
        assert np.linalg.norm(T_high) < np.linalg.norm(T_low)


# ── Surgical fix v2 math tests ──────────────────────────────────────────────


class TestComputePerLayerScales:
    def test_returns_list_per_position(self):
        pruned = [np.random.randn(100, 8) for _ in range(5)]
        target = [np.random.randn(100, 8) for _ in range(5)]
        scales = compute_per_layer_scales(pruned, target)
        assert len(scales) == 5

    def test_identical_scales_are_one(self):
        acts = [np.random.randn(100, 8) for _ in range(3)]
        scales = compute_per_layer_scales(acts, acts)
        for s in scales:
            assert s == pytest.approx(1.0, abs=1e-5)

    def test_scaled_target_recovered(self):
        """If each target = 2×pruned, all scales should be 2."""
        pruned = [np.random.randn(100, 8) for _ in range(3)]
        target = [2.0 * p for p in pruned]
        scales = compute_per_layer_scales(pruned, target)
        for s in scales:
            assert s == pytest.approx(2.0, abs=1e-4)


class TestOrthogonalProcrustes:
    def test_identity_if_same(self):
        """Same matrix should yield identity rotation."""
        np.random.seed(0)
        X = np.random.randn(500, 16)
        R = orthogonal_procrustes(X, X)
        np.testing.assert_allclose(R, np.eye(16), atol=0.01)

    def test_is_orthogonal(self):
        """Result should be orthogonal: R^T R = I."""
        np.random.seed(0)
        X = np.random.randn(200, 16)
        Y = np.random.randn(200, 16)
        R = orthogonal_procrustes(X, Y)
        np.testing.assert_allclose(R.T @ R, np.eye(16), atol=1e-6)
        np.testing.assert_allclose(R @ R.T, np.eye(16), atol=1e-6)

    def test_recovers_known_rotation(self):
        """If Y = X @ R_true, we should recover R_true."""
        np.random.seed(0)
        # Generate a random orthogonal matrix via QR
        A = np.random.randn(16, 16)
        R_true, _ = np.linalg.qr(A)
        X = np.random.randn(500, 16)
        Y = X @ R_true
        R_fit = orthogonal_procrustes(X, Y)
        np.testing.assert_allclose(R_fit, R_true, atol=1e-6)

    def test_shape(self):
        X = np.random.randn(100, 32)
        Y = np.random.randn(100, 32)
        R = orthogonal_procrustes(X, Y)
        assert R.shape == (32, 32)


class TestReducedRankRegression:
    def test_shape(self):
        X = np.random.randn(200, 32)
        Y = np.random.randn(200, 32)
        T = reduced_rank_regression(X, Y, rank=8)
        assert T.shape == (32, 32)

    def test_rank_constraint(self):
        """Fitted T should have rank at most the requested rank."""
        np.random.seed(0)
        X = np.random.randn(500, 16)
        Y = np.random.randn(500, 16)
        T = reduced_rank_regression(X, Y, rank=4, ridge=1e-6)
        _, s, _ = np.linalg.svd(T)
        # Top 4 singular values should be nonzero; rest should be zero
        assert np.all(s[4:] < 1e-8)

    def test_full_rank_matches_ridge(self):
        """Rank=d reduced-rank should match full-rank ridge."""
        np.random.seed(0)
        X = np.random.randn(500, 16)
        Y = np.random.randn(500, 16)
        T_rrr = reduced_rank_regression(X, Y, rank=16, ridge=1e-3)
        T_full = fit_linear_probe(X, Y, ridge=1e-3)
        np.testing.assert_allclose(T_rrr, T_full, atol=1e-8)

    def test_identity_if_same(self):
        """Y=X with rank=d should give ~identity."""
        np.random.seed(0)
        X = np.random.randn(500, 16)
        T = reduced_rank_regression(X, X, rank=16, ridge=1e-6)
        np.testing.assert_allclose(T, np.eye(16), atol=0.01)
