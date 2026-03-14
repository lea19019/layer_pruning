"""IFR-guided pruning: use attribution scores to select layers for removal."""

import json
from pathlib import Path

from src.config import NUM_LAYERS, PRUNE_TARGETS


def select_layers_fixed(
    ranking: list[int],
    n_remove: int,
) -> list[int]:
    """Select a fixed number of layers to remove based on IFR ranking.

    Args:
        ranking: Layer indices sorted by importance (least important first).
        n_remove: Number of layers to remove.

    Returns:
        List of layer indices to remove.
    """
    return sorted(ranking[:n_remove])


def select_layers_threshold(
    importance_scores: list[float],
    threshold_factor: float = 0.5,
) -> list[int]:
    """Select layers to remove based on an importance threshold.

    Removes layers whose importance is below threshold_factor * mean importance.
    This lets the scores determine how many layers to prune rather than a
    fixed target.

    Args:
        importance_scores: Per-layer importance scores.
        threshold_factor: Fraction of mean importance below which layers are pruned.

    Returns:
        List of layer indices to remove.
    """
    mean_importance = sum(importance_scores) / len(importance_scores)
    threshold = threshold_factor * mean_importance

    to_remove = [
        i for i, score in enumerate(importance_scores)
        if score < threshold
    ]

    # Never remove the first or last layer (embedding projection / output)
    to_remove = [i for i in to_remove if 0 < i < len(importance_scores) - 1]

    return sorted(to_remove)


def load_ifr_scores(scores_path: Path) -> dict:
    """Load pre-computed IFR scores from JSON."""
    with open(scores_path) as f:
        return json.load(f)


def get_pruning_plan(
    scores_path: Path,
    n_remove: int | None = None,
    threshold_factor: float | None = None,
) -> list[int]:
    """Get the list of layers to remove from IFR scores.

    Either n_remove (fixed count) or threshold_factor (adaptive) must be provided.
    """
    data = load_ifr_scores(scores_path)
    ranking = data["ranking_least_important_first"]
    importance = data["layer_importance"]

    if n_remove is not None:
        layers = select_layers_fixed(ranking, n_remove)
        print(f"IFR fixed pruning: removing {len(layers)} layers: {layers}")
    elif threshold_factor is not None:
        layers = select_layers_threshold(importance, threshold_factor)
        print(f"IFR threshold pruning (factor={threshold_factor}): "
              f"removing {len(layers)} layers: {layers}")
    else:
        raise ValueError("Must provide either n_remove or threshold_factor")

    return layers
