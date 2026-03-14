"""Tests for src/attribution/ifr.py.

Uses tiny mock models to verify the proximity function and IFRScorer hooks
without downloading any real models.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from src.attribution.ifr import IFRScorer, proximity


# ---------------------------------------------------------------------------
# proximity function
# ---------------------------------------------------------------------------

class TestProximity:
    def test_identical_vectors(self):
        """If component == total, proximity should be positive."""
        t = torch.tensor([[1.0, 2.0, 3.0]])
        score = proximity(t, t)
        # -||t - t||_1 + ||t||_1 = 0 + 6 = 6
        assert score.item() > 0

    def test_zero_component(self):
        """Zero component should have zero proximity (total == rest)."""
        total = torch.tensor([[1.0, 2.0, 3.0]])
        component = torch.zeros_like(total)
        score = proximity(component, total)
        # -||total - 0||_1 + ||total||_1 = -6 + 6 = 0
        assert score.item() == pytest.approx(0.0, abs=1e-6)

    def test_non_negative(self):
        """Proximity is clamped to be non-negative."""
        total = torch.tensor([[1.0, 0.0]])
        component = torch.tensor([[0.0, 10.0]])
        score = proximity(component, total)
        assert score.item() >= 0.0

    def test_batch_dimension(self):
        """Should average across batch and sequence dimensions."""
        component = torch.randn(2, 5, 8)
        total = component + torch.randn(2, 5, 8) * 0.1
        score = proximity(component, total)
        assert score.shape == ()  # scalar
        assert score.item() >= 0.0

    def test_larger_component_contribution(self):
        """Component that dominates total should have higher proximity."""
        total = torch.tensor([[10.0, 10.0]])
        big_component = torch.tensor([[9.0, 9.0]])
        small_component = torch.tensor([[1.0, 1.0]])

        big_score = proximity(big_component, total)
        small_score = proximity(small_component, total)
        assert big_score.item() > small_score.item()

    def test_2d_input(self):
        """Should work with [seq, hidden] shape."""
        component = torch.randn(5, 8)
        total = component + torch.randn(5, 8) * 0.1
        score = proximity(component, total)
        assert score.shape == ()
        assert score.item() >= 0.0


# ---------------------------------------------------------------------------
# IFRScorer with mock model
# ---------------------------------------------------------------------------

class TestIFRScorerWithMock:
    """Test IFRScorer using a tiny mock model injected post-init."""

    def _make_scorer_with_mock(self, mock_model, mock_tokenizer):
        """Create an IFRScorer without loading a real model."""
        with patch.object(IFRScorer, "__init__", lambda self, **kw: None):
            scorer = IFRScorer.__new__(IFRScorer)

        scorer.device = "cpu"
        scorer.dtype = torch.float32
        scorer.model_name = "mock"
        scorer.tokenizer = mock_tokenizer
        scorer.model = mock_model
        scorer.num_layers = mock_model.config.num_hidden_layers
        scorer._activations = {}
        scorer._hooks = []
        return scorer

    def test_hooks_capture_activations(self, mock_model_2layer, mock_tokenizer):
        scorer = self._make_scorer_with_mock(mock_model_2layer, mock_tokenizer)

        # Manually register hooks
        scorer._register_hooks()

        # Run forward pass
        input_ids = torch.tensor([[1, 2, 3]])
        mock_model_2layer(input_ids)

        # Check that activations were captured
        assert "layer_0_input" in scorer._activations
        assert "layer_0_attn_out" in scorer._activations
        assert "layer_0_mlp_out" in scorer._activations
        assert "layer_0_output" in scorer._activations
        assert "layer_1_input" in scorer._activations
        assert "layer_1_attn_out" in scorer._activations
        assert "layer_1_mlp_out" in scorer._activations
        assert "layer_1_output" in scorer._activations

        scorer._clear_hooks()

    def test_score_single_returns_correct_keys(self, mock_model_2layer, mock_tokenizer):
        scorer = self._make_scorer_with_mock(mock_model_2layer, mock_tokenizer)

        scores = scorer.score_single("hello world test")
        assert "attn_importance" in scores
        assert "mlp_importance" in scores
        assert "layer_importance" in scores

    def test_score_single_correct_shape(self, mock_model_2layer, mock_tokenizer):
        scorer = self._make_scorer_with_mock(mock_model_2layer, mock_tokenizer)

        scores = scorer.score_single("hello world test")
        assert scores["attn_importance"].shape == (2,)
        assert scores["mlp_importance"].shape == (2,)
        assert scores["layer_importance"].shape == (2,)

    def test_scores_non_negative(self, mock_model_2layer, mock_tokenizer):
        scorer = self._make_scorer_with_mock(mock_model_2layer, mock_tokenizer)

        scores = scorer.score_single("hello world test")
        assert (scores["attn_importance"] >= 0).all()
        assert (scores["mlp_importance"] >= 0).all()
        assert (scores["layer_importance"] >= 0).all()

    def test_layer_importance_is_sum(self, mock_model_2layer, mock_tokenizer):
        scorer = self._make_scorer_with_mock(mock_model_2layer, mock_tokenizer)

        scores = scorer.score_single("hello world test")
        expected = scores["attn_importance"] + scores["mlp_importance"]
        torch.testing.assert_close(scores["layer_importance"], expected)

    def test_hooks_cleared_after_scoring(self, mock_model_2layer, mock_tokenizer):
        scorer = self._make_scorer_with_mock(mock_model_2layer, mock_tokenizer)

        scorer.score_single("test")
        assert len(scorer._hooks) == 0
        assert len(scorer._activations) == 0

    def test_rank_layers(self, mock_model_2layer, mock_tokenizer):
        scorer = self._make_scorer_with_mock(mock_model_2layer, mock_tokenizer)

        scores = {
            "layer_importance": torch.tensor([0.5, 0.1, 0.9])
        }
        ranking = scorer.rank_layers(scores)
        # Ascending: least important first
        assert ranking[0] == 1  # score 0.1
        assert ranking[-1] == 2  # score 0.9

    def test_score_dataset(self, mock_model_2layer, mock_tokenizer):
        scorer = self._make_scorer_with_mock(mock_model_2layer, mock_tokenizer)

        texts = ["hello world", "foo bar baz"]
        scores = scorer.score_dataset(texts)

        assert scores["layer_importance"].shape == (2,)
        assert (scores["layer_importance"] >= 0).all()
