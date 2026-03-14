"""Evaluation metrics: COMET, chrF++, BLEU, model size, inference speed."""

import time
from pathlib import Path

import sacrebleu
import torch

from src.config import COMET_MODEL

# Cache the COMET model so we don't reload it for every call
_comet_model = None


def _get_comet_model(model_name: str = COMET_MODEL):
    """Load COMET model once and cache it. Works offline if pre-downloaded."""
    global _comet_model
    if _comet_model is not None:
        return _comet_model

    from comet import download_model, load_from_checkpoint

    model_path = download_model(model_name)
    _comet_model = load_from_checkpoint(model_path)
    return _comet_model


def compute_comet(
    hypotheses: list[str],
    references: list[str],
    sources: list[str],
    model_name: str = COMET_MODEL,
    batch_size: int = 32,
    gpus: int = 1,
) -> float:
    """Compute COMET score.

    Returns:
        System-level COMET score.
    """
    model = _get_comet_model(model_name)

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


def compute_model_size(model: torch.nn.Module) -> dict:
    """Compute model parameter count and estimated disk size.

    Returns:
        Dict with 'total_params', 'trainable_params', 'size_mb'.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate disk size based on dtype
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    size_mb = total_bytes / (1024 * 1024)

    return {
        "total_params": total,
        "trainable_params": trainable,
        "size_mb": round(size_mb, 1),
    }


def measure_inference_speed(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 128,
    n_samples: int = 20,
    warmup: int = 3,
) -> dict:
    """Measure inference speed in tokens/second.

    Args:
        model: CausalLM model.
        tokenizer: Tokenizer.
        prompts: List of prompts to benchmark on.
        max_new_tokens: Tokens to generate per prompt.
        n_samples: Number of prompts to time (after warmup).
        warmup: Number of warmup runs.

    Returns:
        Dict with 'tokens_per_second', 'seconds_per_sample'.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    test_prompts = prompts[: warmup + n_samples]

    # Warmup
    for p in test_prompts[:warmup]:
        inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed runs
    total_tokens = 0
    start = time.perf_counter()

    for p in test_prompts[warmup : warmup + n_samples]:
        inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        # Count only newly generated tokens
        total_tokens += output.shape[1] - inputs["input_ids"].shape[1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return {
        "tokens_per_second": round(total_tokens / elapsed, 1),
        "seconds_per_sample": round(elapsed / n_samples, 2),
        "total_tokens": total_tokens,
        "n_samples": n_samples,
    }


def evaluate_all(
    hypotheses: list[str],
    references: list[str],
    sources: list[str],
    model: torch.nn.Module | None = None,
    tokenizer=None,
    prompts: list[str] | None = None,
) -> dict:
    """Compute all metrics, optionally including model size and speed.

    Args:
        hypotheses: Generated translations.
        references: Reference translations.
        sources: Source sentences.
        model: Optional model for size/speed measurement.
        tokenizer: Optional tokenizer for speed measurement.
        prompts: Optional prompts for speed benchmarking.

    Returns:
        Dict with all metric scores.
    """
    results = {
        "comet": compute_comet(hypotheses, references, sources),
        "chrf": compute_chrf(hypotheses, references),
        "bleu": compute_bleu(hypotheses, references),
    }

    if model is not None:
        results["model_size"] = compute_model_size(model)

    if model is not None and tokenizer is not None and prompts is not None:
        results["inference_speed"] = measure_inference_speed(model, tokenizer, prompts)

    print(f"COMET:  {results['comet']:.4f}")
    print(f"chrF++: {results['chrf']:.2f}")
    print(f"BLEU:   {results['bleu']:.2f}")
    if "model_size" in results:
        print(f"Params: {results['model_size']['total_params']:,} ({results['model_size']['size_mb']:.0f} MB)")
    if "inference_speed" in results:
        print(f"Speed:  {results['inference_speed']['tokens_per_second']:.1f} tok/s")

    return results
