"""CLI script to run IFR attribution and save layer importance scores."""

import argparse
import json
from pathlib import Path

import torch

from src.attribution.ifr import IFRScorer
from src.config import (
    BASE_MODEL,
    FILTERED_DIR,
    RESULTS_DIR,
    SRC_LANG_NAME,
    TGT_LANG_NAME,
    TRANSLATION_PROMPT,
)
from src.utils import load_env, set_seed


def prepare_translation_prompts(
    src_path: Path,
    tgt_path: Path,
    n_samples: int = 200,
) -> list[str]:
    """Create translation prompts from parallel data for IFR scoring."""
    with open(src_path) as f:
        sources = f.read().splitlines()
    with open(tgt_path) as f:
        targets = f.read().splitlines()

    # Use the first n_samples pairs
    prompts = []
    for src, tgt in zip(sources[:n_samples], targets[:n_samples]):
        prompt = TRANSLATION_PROMPT.format(
            src_lang=SRC_LANG_NAME,
            tgt_lang=TGT_LANG_NAME,
            source=src,
        )
        # Append the reference target so the model processes the full translation
        prompts.append(prompt + " " + tgt)

    return prompts


def main():
    parser = argparse.ArgumentParser(description="Score layer importance via IFR")
    parser.add_argument("--model", default=BASE_MODEL, help="Model name or path")
    parser.add_argument("--src", type=Path, default=FILTERED_DIR / "test.cs")
    parser.add_argument("--tgt", type=Path, default=FILTERED_DIR / "test.de")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of examples to score on")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "ifr_scores.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    load_env()

    prompts = prepare_translation_prompts(args.src, args.tgt, args.n_samples)
    print(f"Prepared {len(prompts)} prompts for IFR scoring")

    scorer = IFRScorer(model_name=args.model)
    scores = scorer.score_dataset(prompts, max_length=args.max_length)

    # Get ranking
    ranking = scorer.rank_layers(scores)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "model": args.model,
        "n_samples": len(prompts),
        "layer_importance": scores["layer_importance"].tolist(),
        "attn_importance": scores["attn_importance"].tolist(),
        "mlp_importance": scores["mlp_importance"].tolist(),
        "ranking_least_important_first": ranking,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"\nLayer importance (descending):")
    sorted_layers = sorted(
        enumerate(scores["layer_importance"].tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    for layer_idx, score in sorted_layers:
        print(f"  Layer {layer_idx:2d}: {score:.4f}")

    print(f"\nLeast important layers (prune candidates): {ranking[:16]}")


if __name__ == "__main__":
    main()
