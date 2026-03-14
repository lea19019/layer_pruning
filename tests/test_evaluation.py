"""Tests for src/evaluation/metrics.py.

COMET is mocked since it requires downloading a model.
chrF++ and BLEU use sacrebleu directly (lightweight, no download needed).
compute_model_size is tested with a simple nn.Module.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.evaluation.metrics import compute_bleu, compute_chrf, compute_model_size


# ---------------------------------------------------------------------------
# compute_chrf
# ---------------------------------------------------------------------------

class TestComputeChrf:
    def test_perfect_score(self):
        hyps = ["Hallo Welt", "Guten Tag"]
        refs = ["Hallo Welt", "Guten Tag"]
        score = compute_chrf(hyps, refs)
        assert score == pytest.approx(100.0, abs=0.1)

    def test_zero_overlap(self):
        hyps = ["xxxxxxxx"]
        refs = ["Hallo Welt"]
        score = compute_chrf(hyps, refs)
        assert score < 10.0

    def test_partial_match(self):
        hyps = ["Hallo"]
        refs = ["Hallo Welt"]
        score = compute_chrf(hyps, refs)
        assert 0 < score < 100

    def test_returns_float(self):
        score = compute_chrf(["a"], ["b"])
        assert isinstance(score, float)

    def test_empty_hypothesis(self):
        score = compute_chrf([""], ["something"])
        assert score == 0.0


# ---------------------------------------------------------------------------
# compute_bleu
# ---------------------------------------------------------------------------

class TestComputeBleu:
    def test_perfect_score(self):
        hyps = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        score = compute_bleu(hyps, refs)
        assert score == pytest.approx(100.0, abs=0.1)

    def test_no_overlap(self):
        hyps = ["xxxx yyyy zzzz"]
        refs = ["the cat sat on the mat"]
        score = compute_bleu(hyps, refs)
        assert score == pytest.approx(0.0, abs=0.1)

    def test_partial_match(self):
        # With a single short sentence, BLEU brevity penalty can make this 0.
        # Use multiple sentences to get a meaningful score.
        hyps = ["the cat sat on the mat", "hello world today"]
        refs = ["the cat sat on the mat", "hello world today now"]
        score = compute_bleu(hyps, refs)
        assert score > 0

    def test_returns_float(self):
        score = compute_bleu(["a b c"], ["d e f"])
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# compute_model_size
# ---------------------------------------------------------------------------

class TestComputeModelSize:
    def test_basic_model(self):
        model = nn.Linear(10, 5, bias=True)
        result = compute_model_size(model)
        assert "total_params" in result
        assert "trainable_params" in result
        assert "size_mb" in result
        # Linear(10, 5, bias=True) has 10*5 + 5 = 55 params
        assert result["total_params"] == 55
        assert result["trainable_params"] == 55

    def test_frozen_params(self):
        model = nn.Linear(10, 5)
        for p in model.parameters():
            p.requires_grad = False
        result = compute_model_size(model)
        assert result["trainable_params"] == 0
        assert result["total_params"] > 0

    def test_size_mb_reasonable(self):
        model = nn.Linear(1000, 1000, bias=False)
        result = compute_model_size(model)
        # 1M float32 params = 4 MB
        assert 3.5 < result["size_mb"] < 4.5

    def test_mock_causal_lm(self, mock_model):
        result = compute_model_size(mock_model)
        assert result["total_params"] > 0
        # Mock model is tiny (~2KB), size_mb may round to 0.0
        assert result["size_mb"] >= 0

    def test_empty_model(self):
        model = nn.Module()
        result = compute_model_size(model)
        assert result["total_params"] == 0
        assert result["size_mb"] == 0.0


# ---------------------------------------------------------------------------
# compute_comet (mocked)
# ---------------------------------------------------------------------------

class TestComputeCometMocked:
    @patch("src.evaluation.metrics._get_comet_model")
    def test_comet_returns_float(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = MagicMock(system_score=0.85)
        mock_get_model.return_value = mock_model

        from src.evaluation.metrics import compute_comet

        score = compute_comet(
            hypotheses=["Hallo Welt"],
            references=["Hallo Welt"],
            sources=["Ahoj svete"],
        )
        assert score == pytest.approx(0.85)
        mock_model.predict.assert_called_once()

    @patch("src.evaluation.metrics._get_comet_model")
    def test_comet_data_format(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = MagicMock(system_score=0.9)
        mock_get_model.return_value = mock_model

        from src.evaluation.metrics import compute_comet

        compute_comet(
            hypotheses=["h1", "h2"],
            references=["r1", "r2"],
            sources=["s1", "s2"],
        )

        call_args = mock_model.predict.call_args
        data = call_args[0][0]
        assert len(data) == 2
        assert data[0] == {"src": "s1", "mt": "h1", "ref": "r1"}
