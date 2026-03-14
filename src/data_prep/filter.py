"""Data filtering pipeline following Moslem et al.

Steps:
1. Deduplication
2. Max 200 words per segment
3. Length ratio ≤ 1.5x
4. Language detection via fastText (threshold 0.9)
5. Semantic similarity via sentence-transformers (threshold 0.7)
"""

import csv
from pathlib import Path

import fasttext
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import (
    FILTERED_DIR,
    LANG_DETECT_THRESHOLD,
    LENGTH_RATIO_MAX,
    MAX_WORDS_PER_SEGMENT,
    RAW_DIR,
    SEMANTIC_MODEL,
    SEMANTIC_SIM_THRESHOLD,
)

# Suppress fasttext warnings about loading with warning
fasttext.FastText.eprint = lambda x: None

FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_MODEL_PATH = RAW_DIR / "lid.176.bin"


def load_raw_pairs(tsv_path: Path) -> list[tuple[str, str]]:
    """Load tab-separated parallel pairs."""
    pairs = []
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                src, tgt = row[0].strip(), row[1].strip()
                if src and tgt:
                    pairs.append((src, tgt))
    print(f"Loaded {len(pairs)} raw pairs")
    return pairs


def dedup(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove exact duplicate pairs."""
    seen = set()
    result = []
    for src, tgt in pairs:
        key = (src, tgt)
        if key not in seen:
            seen.add(key)
            result.append((src, tgt))
    print(f"After dedup: {len(result)} pairs (removed {len(pairs) - len(result)})")
    return result


def filter_length(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove segments exceeding max words or with extreme length ratios."""
    result = []
    for src, tgt in pairs:
        src_words = len(src.split())
        tgt_words = len(tgt.split())
        if src_words > MAX_WORDS_PER_SEGMENT or tgt_words > MAX_WORDS_PER_SEGMENT:
            continue
        if src_words == 0 or tgt_words == 0:
            continue
        ratio = max(src_words, tgt_words) / min(src_words, tgt_words)
        if ratio > LENGTH_RATIO_MAX:
            continue
        result.append((src, tgt))
    print(f"After length filter: {len(result)} pairs (removed {len(pairs) - len(result)})")
    return result


def _ensure_fasttext_model() -> Path:
    """Download fastText language ID model if needed."""
    if not FASTTEXT_MODEL_PATH.exists():
        import urllib.request
        print(f"Downloading fastText LID model ...")
        FASTTEXT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FASTTEXT_MODEL_URL, FASTTEXT_MODEL_PATH)
    return FASTTEXT_MODEL_PATH


def filter_language(
    pairs: list[tuple[str, str]],
    src_lang: str = "cs",
    tgt_lang: str = "de",
) -> list[tuple[str, str]]:
    """Filter pairs where detected language doesn't match expected."""
    model_path = _ensure_fasttext_model()
    model = fasttext.load_model(str(model_path))

    result = []
    for src, tgt in tqdm(pairs, desc="Language detection"):
        # fastText predict() returns (('__label__xx',), array([confidence])).
        # The "__label__" prefix is fastText's convention for supervised labels;
        # we strip it to get the ISO 639-1 language code (e.g. "cs", "de").
        src_pred = model.predict(src.replace("\n", " "))
        tgt_pred = model.predict(tgt.replace("\n", " "))

        src_label = src_pred[0][0].replace("__label__", "")
        src_score = src_pred[1][0]
        tgt_label = tgt_pred[0][0].replace("__label__", "")
        tgt_score = tgt_pred[1][0]

        if (
            src_label == src_lang
            and src_score >= LANG_DETECT_THRESHOLD
            and tgt_label == tgt_lang
            and tgt_score >= LANG_DETECT_THRESHOLD
        ):
            result.append((src, tgt))

    print(f"After lang filter: {len(result)} pairs (removed {len(pairs) - len(result)})")
    return result


def filter_semantic_similarity(
    pairs: list[tuple[str, str]],
    batch_size: int = 256,
) -> list[tuple[str, str]]:
    """Keep pairs with cosine similarity >= threshold using multilingual embeddings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(SEMANTIC_MODEL, device=device)

    sources = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]

    print("Encoding source sentences ...")
    src_embs = model.encode(sources, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    print("Encoding target sentences ...")
    tgt_embs = model.encode(targets, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

    # Dot product equals cosine similarity since embeddings are L2-normalized.
    similarities = np.sum(src_embs * tgt_embs, axis=1)

    result = [
        pair for pair, sim in zip(pairs, similarities)
        if sim >= SEMANTIC_SIM_THRESHOLD
    ]
    print(f"After semantic filter: {len(result)} pairs (removed {len(pairs) - len(result)})")
    return result


def run_full_pipeline(tsv_path: Path | None = None, skip_semantic: bool = False) -> list[tuple[str, str]]:
    """Run the complete filtering pipeline."""
    if tsv_path is None:
        tsv_path = RAW_DIR / "news-commentary-v18.cs-de.tsv"

    # Filters are ordered cheapest-first: dedup and length checks are O(n)
    # string ops, language detection loads a small fastText model, and
    # semantic similarity requires GPU-based sentence-transformer encoding.
    # Each stage reduces the dataset so later expensive stages process fewer pairs.
    pairs = load_raw_pairs(tsv_path)
    pairs = dedup(pairs)
    pairs = filter_length(pairs)
    pairs = filter_language(pairs)
    if not skip_semantic:
        pairs = filter_semantic_similarity(pairs)

    return pairs


def save_pairs(pairs: list[tuple[str, str]], output_dir: Path, prefix: str = "nc18"):
    """Save filtered pairs as parallel text files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    src_path = output_dir / f"{prefix}.cs"
    tgt_path = output_dir / f"{prefix}.de"

    with open(src_path, "w", encoding="utf-8") as f_src, \
         open(tgt_path, "w", encoding="utf-8") as f_tgt:
        for src, tgt in pairs:
            f_src.write(src + "\n")
            f_tgt.write(tgt + "\n")

    print(f"Saved {len(pairs)} pairs to {output_dir}")
    return src_path, tgt_path


if __name__ == "__main__":
    from src.data_prep.download import download_corpus

    tsv = download_corpus()
    pairs = run_full_pipeline(tsv)
    save_pairs(pairs, FILTERED_DIR)
