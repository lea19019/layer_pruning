"""Experiment 2: Categorize errors in pruned model outputs.

Classifies translations from IP_16_enes (pruned, no FT) into error types:
  - wrong_language: output is not Spanish
  - hallucination: output contains content not in source
  - repetition: repeated phrases/sentences
  - truncation: output is much shorter than reference
  - verbose: output is much longer than reference
  - partial: some content translated correctly, some missing
  - garbage: unintelligible output

Also compares with I2_16_enes (pruned+FT+KD) and B4_enes (full+FT+KD)
to show what fine-tuning fixes.

Usage:
    python -m ablation.scripts.output_categorization [--results-dir experiments/results]
"""

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "ablation/results"


def detect_language_heuristic(text: str) -> str:
    """Simple heuristic language detection based on common words."""
    text_lower = text.lower()
    spanish_markers = [
        "el", "la", "los", "las", "de", "en", "que", "es", "un", "una",
        "por", "con", "para", "como", "pero", "del", "al", "se",
    ]
    english_markers = [
        "the", "is", "are", "was", "were", "have", "has", "been", "will",
        "would", "could", "should", "this", "that", "with", "from",
    ]
    words = set(re.findall(r"\b\w+\b", text_lower))
    es_count = sum(1 for w in spanish_markers if w in words)
    en_count = sum(1 for w in english_markers if w in words)

    if es_count > en_count + 2:
        return "es"
    elif en_count > es_count + 2:
        return "en"
    return "unknown"


def detect_repetition(text: str, min_repeat_len: int = 10) -> float:
    """Detect repeated substrings. Returns fraction of text that is repeated."""
    if len(text) < min_repeat_len * 2:
        return 0.0

    # Check for repeated phrases (ngrams)
    words = text.split()
    if len(words) < 4:
        return 0.0

    # Bigram repetition
    bigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    if not bigrams:
        return 0.0
    unique_ratio = len(set(bigrams)) / len(bigrams)
    return 1.0 - unique_ratio


def categorize_translation(
    source: str, hypothesis: str, reference: str
) -> dict:
    """Categorize a single translation's error type.

    Returns dict with:
        - categories: list of error category strings
        - details: dict of metrics
    """
    categories = []
    details = {}

    # Empty or near-empty
    if len(hypothesis.strip()) < 5:
        categories.append("garbage")
        details["reason"] = "near-empty output"
        return {"categories": categories, "details": details}

    # Language detection
    lang = detect_language_heuristic(hypothesis)
    details["detected_lang"] = lang
    if lang == "en":
        categories.append("wrong_language")

    # Length ratio analysis
    ref_words = len(reference.split())
    hyp_words = len(hypothesis.split())
    src_words = len(source.split())
    if ref_words > 0:
        length_ratio = hyp_words / ref_words
    else:
        length_ratio = float("inf")
    details["length_ratio_vs_ref"] = round(length_ratio, 2)
    details["hyp_words"] = hyp_words
    details["ref_words"] = ref_words
    details["src_words"] = src_words

    if length_ratio < 0.3:
        categories.append("truncation")
    elif length_ratio > 3.0:
        categories.append("verbose")

    # Repetition detection
    rep_score = detect_repetition(hypothesis)
    details["repetition_score"] = round(rep_score, 3)
    if rep_score > 0.5:
        categories.append("repetition")

    # Source copying (hypothesis contains too much of the source verbatim)
    src_lower = source.lower()
    hyp_lower = hypothesis.lower()
    src_words_set = set(src_lower.split())
    hyp_words_set = set(hyp_lower.split())
    if src_words_set and len(src_words_set) > 3:
        overlap = len(src_words_set & hyp_words_set) / len(src_words_set)
        details["source_overlap"] = round(overlap, 3)
        if overlap > 0.7 and lang != "es":
            categories.append("source_copy")

    if not categories:
        # Check if it looks like a reasonable translation
        if lang == "es" or lang == "unknown":
            categories.append("plausible")
        else:
            categories.append("unclear")

    return {"categories": categories, "details": details}


def load_translations(results_json_path: Path) -> list[dict]:
    """Load sample translations from a results.json file."""
    with open(results_json_path) as f:
        data = json.load(f)
    return data.get("sample_translations", [])


def load_full_translations(result_dir: Path) -> list[dict] | None:
    """Load full translation outputs if available."""
    hyp_path = result_dir / "translations.txt"
    src_path = PROJECT_ROOT / "data/filtered_en_es/test.en"
    ref_path = PROJECT_ROOT / "data/filtered_en_es/test.es"

    if not hyp_path.exists():
        return None

    with open(hyp_path) as f:
        hypotheses = [line.strip() for line in f]
    with open(src_path) as f:
        sources = [line.strip() for line in f]
    with open(ref_path) as f:
        references = [line.strip() for line in f]

    n = min(len(hypotheses), len(sources), len(references))
    return [
        {"source": sources[i], "hypothesis": hypotheses[i], "reference": references[i]}
        for i in range(n)
    ]


def analyze_model(name: str, translations: list[dict]) -> dict:
    """Run categorization on all translations for a model."""
    results = []
    category_counts = {}

    for t in translations:
        cat = categorize_translation(t["source"], t["hypothesis"], t["reference"])
        cat["source"] = t["source"][:100]
        cat["hypothesis"] = t["hypothesis"][:200]
        cat["reference"] = t["reference"][:100]
        results.append(cat)
        for c in cat["categories"]:
            category_counts[c] = category_counts.get(c, 0) + 1

    total = len(translations)
    summary = {
        "model": name,
        "total_translations": total,
        "category_counts": category_counts,
        "category_pcts": {
            k: round(v / total * 100, 1) for k, v in category_counts.items()
        },
        "avg_length_ratio": round(
            sum(r["details"]["length_ratio_vs_ref"] for r in results) / max(total, 1),
            2,
        ),
        "avg_repetition": round(
            sum(r["details"]["repetition_score"] for r in results) / max(total, 1),
            3,
        ),
    }
    return {"summary": summary, "per_sentence": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/results",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models_to_analyze = {
        "IP_16_enes": args.results_dir / "IP_16_enes",
        "I2_16_enes": args.results_dir / "I2_16_enes",
        "B4_enes": args.results_dir / "B4_enes",
    }

    all_results = {}
    for name, result_dir in models_to_analyze.items():
        results_json = result_dir / "results.json"
        if not results_json.exists():
            print(f"Skipping {name}: {results_json} not found")
            continue

        print(f"\n=== Analyzing {name} ===")

        # Try full translations first, fall back to samples
        translations = load_full_translations(result_dir)
        if translations:
            print(f"  Using full translations ({len(translations)} sentences)")
        else:
            translations = load_translations(results_json)
            print(f"  Using sample translations ({len(translations)} sentences)")

        analysis = analyze_model(name, translations)
        all_results[name] = analysis

        print(f"  Category distribution:")
        for cat, count in sorted(
            analysis["summary"]["category_counts"].items(),
            key=lambda x: -x[1],
        ):
            pct = analysis["summary"]["category_pcts"][cat]
            print(f"    {cat}: {count} ({pct}%)")
        print(f"  Avg length ratio vs ref: {analysis['summary']['avg_length_ratio']}")
        print(f"  Avg repetition score: {analysis['summary']['avg_repetition']}")

    # Save results
    output_path = RESULTS_DIR / "output_categorization.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # Print comparison table
    if len(all_results) > 1:
        print("\n=== Comparison ===")
        all_cats = set()
        for r in all_results.values():
            all_cats.update(r["summary"]["category_counts"].keys())

        header = f"{'Category':<20}" + "".join(
            f"{name:<20}" for name in all_results.keys()
        )
        print(header)
        print("-" * len(header))
        for cat in sorted(all_cats):
            row = f"{cat:<20}"
            for name in all_results:
                pct = all_results[name]["summary"]["category_pcts"].get(cat, 0)
                row += f"{pct:>6.1f}%             "
            print(row)


if __name__ == "__main__":
    main()
