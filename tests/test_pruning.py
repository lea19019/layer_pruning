"""Tests for src/pruning/remove_layers.py and src/pruning/guided.py."""

import json

import pytest
import torch

from src.pruning.guided import (
    get_pruning_plan,
    select_layers_fixed,
    select_layers_threshold,
)
from src.pruning.remove_layers import remove_layers


# ---------------------------------------------------------------------------
# remove_layers
# ---------------------------------------------------------------------------

class TestRemoveLayers:
    def test_remove_single_layer(self, mock_model):
        original_count = len(mock_model.model.layers)
        remove_layers(mock_model, [1])
        assert len(mock_model.model.layers) == original_count - 1
        assert mock_model.config.num_hidden_layers == original_count - 1

    def test_remove_multiple_layers(self, mock_model):
        remove_layers(mock_model, [0, 2])
        assert len(mock_model.model.layers) == 2
        assert mock_model.config.num_hidden_layers == 2

    def test_layer_idx_reindexed(self, mock_model):
        """After removal, layer_idx should be 0, 1, 2, ..."""
        remove_layers(mock_model, [1])
        for i, layer in enumerate(mock_model.model.layers):
            assert layer.self_attn.layer_idx == i

    def test_layer_idx_reindexed_after_multiple_removal(self, mock_model):
        remove_layers(mock_model, [0, 2])
        for i, layer in enumerate(mock_model.model.layers):
            assert layer.self_attn.layer_idx == i

    def test_config_updated(self, mock_model):
        remove_layers(mock_model, [0, 1, 2])
        assert mock_model.config.num_hidden_layers == 1

    def test_out_of_range_raises(self, mock_model):
        with pytest.raises(ValueError, match="out of range"):
            remove_layers(mock_model, [10])

    def test_negative_index_raises(self, mock_model):
        with pytest.raises(ValueError, match="out of range"):
            remove_layers(mock_model, [-1])

    def test_remove_all_but_one(self, mock_model):
        remove_layers(mock_model, [0, 1, 2])
        assert len(mock_model.model.layers) == 1
        assert mock_model.model.layers[0].self_attn.layer_idx == 0

    def test_model_still_runs_after_pruning(self, mock_model):
        remove_layers(mock_model, [1, 2])
        input_ids = torch.tensor([[1, 2, 3]])
        output = mock_model(input_ids)
        assert output.logits is not None

    def test_unsorted_removal_order(self, mock_model):
        """remove_layers should handle unsorted input correctly."""
        remove_layers(mock_model, [2, 0])
        assert len(mock_model.model.layers) == 2
        for i, layer in enumerate(mock_model.model.layers):
            assert layer.self_attn.layer_idx == i


# ---------------------------------------------------------------------------
# select_layers_fixed
# ---------------------------------------------------------------------------

class TestSelectLayersFixed:
    def test_selects_n_least_important(self):
        ranking = [5, 3, 1, 0, 2, 4]  # least important first
        result = select_layers_fixed(ranking, n_remove=3)
        assert len(result) == 3
        assert result == sorted(result)  # should be sorted
        assert set(result) == {1, 3, 5}

    def test_remove_zero(self):
        ranking = [1, 2, 3]
        result = select_layers_fixed(ranking, n_remove=0)
        assert result == []

    def test_remove_all(self):
        ranking = [2, 0, 1]
        result = select_layers_fixed(ranking, n_remove=3)
        assert len(result) == 3
        assert sorted(result) == [0, 1, 2]

    def test_result_sorted(self):
        ranking = [7, 3, 1, 5, 2, 6, 0, 4]
        result = select_layers_fixed(ranking, n_remove=4)
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# select_layers_threshold
# ---------------------------------------------------------------------------

class TestSelectLayersThreshold:
    def test_removes_below_threshold(self):
        scores = [1.0, 0.1, 0.1, 1.0]
        # mean = 0.55, threshold = 0.5 * 0.55 = 0.275
        result = select_layers_threshold(scores, threshold_factor=0.5)
        assert 1 in result
        assert 2 in result

    def test_never_removes_first_layer(self):
        scores = [0.01, 0.5, 0.5, 0.5]
        result = select_layers_threshold(scores, threshold_factor=0.9)
        assert 0 not in result

    def test_never_removes_last_layer(self):
        scores = [0.5, 0.5, 0.5, 0.01]
        result = select_layers_threshold(scores, threshold_factor=0.9)
        assert 3 not in result

    def test_all_equal_scores(self):
        scores = [0.5, 0.5, 0.5, 0.5]
        # threshold = 0.5 * 0.5 = 0.25; all above threshold
        result = select_layers_threshold(scores, threshold_factor=0.5)
        assert result == []

    def test_high_threshold_removes_more(self):
        scores = [1.0, 0.3, 0.4, 0.2, 1.0]
        low_result = select_layers_threshold(scores, threshold_factor=0.3)
        high_result = select_layers_threshold(scores, threshold_factor=0.9)
        assert len(high_result) >= len(low_result)

    def test_result_sorted(self):
        scores = [1.0, 0.1, 0.05, 0.2, 1.0]
        result = select_layers_threshold(scores, threshold_factor=0.5)
        assert result == sorted(result)

    def test_zero_threshold(self):
        """With threshold_factor=0, no scores can be below 0, so nothing removed."""
        scores = [0.1, 0.2, 0.3, 0.4]
        result = select_layers_threshold(scores, threshold_factor=0.0)
        assert result == []


# ---------------------------------------------------------------------------
# get_pruning_plan
# ---------------------------------------------------------------------------

class TestGetPruningPlan:
    def test_fixed_count(self, sample_ifr_scores):
        layers = get_pruning_plan(sample_ifr_scores, n_remove=2)
        assert len(layers) == 2
        assert layers == sorted(layers)

    def test_threshold_based(self, sample_ifr_scores):
        layers = get_pruning_plan(sample_ifr_scores, threshold_factor=0.5)
        assert isinstance(layers, list)
        assert all(isinstance(i, int) for i in layers)

    def test_must_provide_argument(self, sample_ifr_scores):
        with pytest.raises(ValueError, match="Must provide"):
            get_pruning_plan(sample_ifr_scores)

    def test_loads_scores_file(self, sample_ifr_scores):
        """Verify it actually reads the JSON file."""
        layers = get_pruning_plan(sample_ifr_scores, n_remove=1)
        # ranking_least_important_first = [1, 2, 0, 3], so first = layer 1
        assert 1 in layers


# ---------------------------------------------------------------------------
# heuristic pruning language params
# ---------------------------------------------------------------------------

class TestHeuristicLanguageParams:
    """Verify heuristic pruning passes language names to prompts."""

    def test_evaluate_without_layer_accepts_lang_params(self):
        """evaluate_without_layer should accept src_lang/tgt_lang kwargs."""
        import inspect
        from src.pruning.heuristic import evaluate_without_layer

        sig = inspect.signature(evaluate_without_layer)
        assert "src_lang" in sig.parameters
        assert "tgt_lang" in sig.parameters

    def test_iterative_prune_accepts_lang_params(self):
        """iterative_prune should accept src_lang/tgt_lang kwargs."""
        import inspect
        from src.pruning.heuristic import iterative_prune

        sig = inspect.signature(iterative_prune)
        assert "src_lang" in sig.parameters
        assert "tgt_lang" in sig.parameters

    def test_evaluate_without_layer_default_is_czech_german(self):
        """Default language params should be Czech/German for backward compat."""
        import inspect
        from src.pruning.heuristic import evaluate_without_layer

        sig = inspect.signature(evaluate_without_layer)
        assert sig.parameters["src_lang"].default == "Czech"
        assert sig.parameters["tgt_lang"].default == "German"

    def test_iterative_prune_default_is_czech_german(self):
        """Default language params should be Czech/German for backward compat."""
        import inspect
        from src.pruning.heuristic import iterative_prune

        sig = inspect.signature(iterative_prune)
        assert sig.parameters["src_lang"].default == "Czech"
        assert sig.parameters["tgt_lang"].default == "German"


# ---------------------------------------------------------------------------
# run_experiment: pruned model reuse
# ---------------------------------------------------------------------------

class TestPrunedModelReuse:
    """Verify run_experiment skips pruning when pruned_model already exists."""

    def test_heuristic_skip_check_in_source(self):
        """run_pipeline should check for existing pruned_model dir."""
        import inspect
        from src.run_experiment import run_pipeline
        source = inspect.getsource(run_pipeline)
        assert "pruned_model" in source
        assert "skipping pruning" in source.lower() or "skip" in source.lower()
