#!/usr/bin/env python3
"""Filter and split English-Spanish News Commentary v18 data.

Applies the same filtering pipeline as Czech-German:
dedup, length filter, language detection, semantic similarity.
Saves to data/filtered_en_es/.

Run on login node (needs internet for fastText model download).
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Some en-es entries are very long, increase CSV field limit
csv.field_size_limit(1_000_000)

from src.data_prep.filter import (
    dedup,
    filter_language,
    filter_length,
    filter_semantic_similarity,
    load_raw_pairs,
)
from src.data_prep.split import save_split, split_data


def main():
    tsv_path = Path("data/raw/news-commentary-v18.en-es.tsv")
    output_dir = Path("data/filtered_en_es")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Filtering English-Spanish News Commentary v18 ===")
    pairs = load_raw_pairs(tsv_path)
    pairs = dedup(pairs)
    pairs = filter_length(pairs)
    pairs = filter_language(pairs, src_lang="en", tgt_lang="es")
    pairs = filter_semantic_similarity(pairs)

    print(f"\nFinal filtered: {len(pairs)} pairs")

    # Split: 100K train, 500 test (same as cs-de)
    train, test = split_data(pairs, train_size=100_000, test_size=500, seed=42)
    save_split(train, output_dir, "train")
    save_split(test, output_dir, "test")

    print(f"\nSaved to {output_dir}:")
    print(f"  train.en: {len(train)} pairs")
    print(f"  test.en: {len(test)} pairs")
    print("Done.")


if __name__ == "__main__":
    main()
