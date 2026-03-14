"""Evaluation metrics: COMET, chrF++, BLEU."""

from pathlib import Path

import sacrebleu
from comet import download_model, load_from_checkpoint

from src.config import COMET_MODEL


def compute_comet(
    hypotheses: list[str],
    references: list[str],
    sources: list[str],
    model_name: str = COMET_MODEL,
    batch_size: int = 32,
    gpus: int = 1,
) -> float:
    """Compute COMET score.

    Args:
        hypotheses: System translations.
        references: Reference translations.
        sources: Source sentences.
        model_name: COMET model name.
        batch_size: Batch size.
        gpus: Number of GPUs.

    Returns:
        System-level COMET score.
    """
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]

    output = model.predict(data, batch_size=batch_size, gpus=gpus)
    return output.system_score


def compute_chrf(
    hypotheses: list[str],
    references: list[str],
) -> float:
    """Compute chrF++ score using sacrebleu."""
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    return chrf.score


def compute_bleu(
    hypotheses: list[str],
    references: list[str],
) -> float:
    """Compute BLEU score using sacrebleu."""
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


def evaluate_all(
    hypotheses: list[str],
    references: list[str],
    sources: list[str],
) -> dict[str, float]:
    """Compute all metrics."""
    results = {
        "comet": compute_comet(hypotheses, references, sources),
        "chrf": compute_chrf(hypotheses, references),
        "bleu": compute_bleu(hypotheses, references),
    }
    print(f"COMET: {results['comet']:.4f}")
    print(f"chrF++: {results['chrf']:.2f}")
    print(f"BLEU: {results['bleu']:.2f}")
    return results
