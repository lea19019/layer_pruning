"""Split filtered data into train/test sets."""

import random
from pathlib import Path

from src.config import FILTERED_DIR, TEST_SIZE, TRAIN_SIZE


def load_parallel(src_path: Path, tgt_path: Path) -> list[tuple[str, str]]:
    """Load parallel text files."""
    with open(src_path) as f_src, open(tgt_path) as f_tgt:
        pairs = list(zip(f_src.read().splitlines(), f_tgt.read().splitlines()))
    return pairs


def split_data(
    pairs: list[tuple[str, str]],
    train_size: int = TRAIN_SIZE,
    test_size: int = TEST_SIZE,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Randomly split pairs into train and test sets."""
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    total_needed = train_size + test_size
    if len(shuffled) < total_needed:
        print(
            f"Warning: only {len(shuffled)} pairs available, "
            f"need {total_needed}. Using all data."
        )
        test = shuffled[:test_size]
        train = shuffled[test_size:]
    else:
        test = shuffled[:test_size]
        train = shuffled[test_size : test_size + train_size]

    print(f"Split: {len(train)} train, {len(test)} test")
    return train, test


def save_split(pairs: list[tuple[str, str]], output_dir: Path, name: str,
               src_ext: str = "cs", tgt_ext: str = "de"):
    """Save a split as parallel files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    src_path = output_dir / f"{name}.{src_ext}"
    tgt_path = output_dir / f"{name}.{tgt_ext}"

    with open(src_path, "w", encoding="utf-8") as f_src, \
         open(tgt_path, "w", encoding="utf-8") as f_tgt:
        for src, tgt in pairs:
            f_src.write(src + "\n")
            f_tgt.write(tgt + "\n")

    print(f"Saved {name}: {len(pairs)} pairs to {output_dir}")


if __name__ == "__main__":
    pairs = load_parallel(FILTERED_DIR / "nc18.cs", FILTERED_DIR / "nc18.de")
    train, test = split_data(pairs)
    save_split(train, FILTERED_DIR, "train")
    save_split(test, FILTERED_DIR, "test")
