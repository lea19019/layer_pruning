#!/usr/bin/env python3
"""Benchmark inference speed using vLLM with fp16 and GPTQ models.

Usage:
    # Test single experiment
    python scripts/benchmark_speed.py --experiment I1_8

    # All experiments
    python scripts/benchmark_speed.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FILTERED_DIR, TRANSLATION_PROMPT
from src.run_experiment import LANG_DEFAULTS, _data_dir, _resolve_lang_pair
from src.utils import load_env

RESULTS_DIR = Path("experiments/results")
N_SAMPLES = 50
WARMUP = 5
MAX_NEW_TOKENS = 128


def get_fp16_model_path(exp_dir: Path) -> str | None:
    """Find the fp16 model for an experiment.

    For quantized experiments (I3, I4, M3, M4), find the corresponding
    non-quantized experiment's finetuned model (same pruning + FT, no quant).
    """
    results_path = exp_dir / "results.json"
    if not results_path.exists():
        return None

    # Check if this experiment has a finetuned model directly
    merged = exp_dir / "finetuned" / "merged"
    if merged.exists() and (merged / "config.json").exists():
        return str(merged)

    # For quantized experiments, find the equivalent non-quantized one
    with open(results_path) as f:
        d = json.load(f)
    cfg = d.get("config", {})

    if not cfg.get("quantization", {}).get("enabled", False):
        # No quantization and no finetuned dir — must be B0 (base model)
        if not cfg.get("finetuning", {}).get("enabled", False):
            return cfg.get("base_model", "CohereForAI/aya-expanse-8b")
        return None

    # Quantized experiment — find the fp16 equivalent
    # I3_8_int8 -> I1_8, I4_8_int8 -> I2_8, M3_8_int8 -> M1_8, etc.
    eid = cfg["experiment_id"]
    pruning = cfg.get("pruning", {})
    method = pruning.get("method", "none")
    has_kd = cfg.get("distillation", {}).get("enabled", False)

    if method == "ifr":
        base_group = "I2" if has_kd else "I1"
        n_remove = pruning.get("n_remove")
        if n_remove:
            equiv = "{}_{}".format(base_group, n_remove)
        elif pruning.get("threshold_factor"):
            equiv = "I5_threshold"
        else:
            return None
    elif method == "heuristic":
        base_group = "M2" if has_kd else "M1"
        target = pruning.get("target_layers")
        n_remove = 32 - target if target else None
        if n_remove:
            equiv = "{}_{}".format(base_group, n_remove)
        else:
            return None
    elif method == "none":
        # B1 -> B0 base model
        return cfg.get("base_model", "CohereForAI/aya-expanse-8b")
    else:
        return None

    # Add _enes suffix if needed
    if "_enes" in eid:
        equiv += "_enes"

    equiv_merged = RESULTS_DIR / equiv / "finetuned" / "merged"
    if equiv_merged.exists() and (equiv_merged / "config.json").exists():
        return str(equiv_merged)

    return None


def build_prompts(exp_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """Build translation prompts, sources, and references from test data."""
    with open(exp_dir / "results.json") as f:
        d = json.load(f)
    cfg = d.get("config", {})
    lang = _resolve_lang_pair(cfg)
    data_dir = _data_dir(lang)

    with open(data_dir / "test.{}".format(lang["src_code"])) as f:
        sources = f.read().splitlines()
    with open(data_dir / "test.{}".format(lang["tgt_code"])) as f:
        references = f.read().splitlines()

    prompts = [
        TRANSLATION_PROMPT.format(
            src_lang=lang["src_name"], tgt_lang=lang["tgt_name"], source=src
        )
        for src in sources
    ]

    return prompts, sources, references


def export_gptq(fp16_model_path: str, output_dir: Path, bits: int = 4) -> str:
    """Export a fp16 model to GPTQ format."""
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    gptq_path = output_dir / "gptq_{}bit".format(bits)

    if gptq_path.exists() and (gptq_path / "config.json").exists():
        print("  GPTQ model already exists at {}".format(gptq_path))
        return str(gptq_path)

    print("  Exporting to GPTQ {}bit: {} -> {}".format(bits, fp16_model_path, gptq_path))

    tokenizer = AutoTokenizer.from_pretrained(fp16_model_path)
    quantize_config = BaseQuantizeConfig(bits=bits, group_size=128, desc_act=False)

    model = AutoGPTQForCausalLM.from_pretrained(
        fp16_model_path, quantize_config=quantize_config
    )

    # Calibration data — a few translation examples
    from src.config import SRC_LANG_NAME, TGT_LANG_NAME
    with open(FILTERED_DIR / "test.cs") as f:
        sources = f.read().splitlines()[:32]
    calib_data = [
        TRANSLATION_PROMPT.format(src_lang=SRC_LANG_NAME, tgt_lang=TGT_LANG_NAME, source=s)
        for s in sources
    ]
    calib_tokens = [tokenizer(t, return_tensors="pt") for t in calib_data]

    model.quantize(calib_tokens)
    model.save_quantized(str(gptq_path))
    tokenizer.save_pretrained(str(gptq_path))

    print("  GPTQ export done: {}".format(gptq_path))
    return str(gptq_path)


def benchmark_vllm(model_path: str, prompts: list[str], label: str = "",
                    evaluate_quality: bool = False, references: list[str] = None,
                    sources: list[str] = None) -> dict:
    """Benchmark with vLLM. Optionally evaluate translation quality."""
    from vllm import LLM, SamplingParams
    from src.evaluation.translate import _extract_translation, STOP_STRINGS
    import torch

    llm = LLM(model=model_path, dtype="auto", trust_remote_code=True,
              max_model_len=768)
    params = SamplingParams(max_tokens=256, temperature=0.0, stop=STOP_STRINGS)

    # Speed benchmark: warmup + timed samples
    test_prompts = prompts[:WARMUP + N_SAMPLES]

    # Warmup
    llm.generate(test_prompts[:WARMUP], params)

    # Timed
    start = time.perf_counter()
    outputs = llm.generate(test_prompts[WARMUP:WARMUP + N_SAMPLES], params)
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    result = {
        "tokens_per_second": round(total_tokens / elapsed, 1),
        "seconds_per_sample": round(elapsed / N_SAMPLES, 2),
        "total_tokens": total_tokens,
        "n_samples": N_SAMPLES,
    }

    print("  {}: {:.1f} tok/s, {:.2f} s/sample".format(
        label, result["tokens_per_second"], result["seconds_per_sample"]))

    # Quality evaluation: translate ALL test sentences
    if evaluate_quality and references and sources:
        from src.evaluation.metrics import compute_comet, compute_chrf, compute_bleu

        all_outputs = llm.generate(prompts, params)
        hypotheses = [_extract_translation(o.outputs[0].text) for o in all_outputs]

        comet = compute_comet(hypotheses, references, sources)
        chrf = compute_chrf(hypotheses, references)
        bleu = compute_bleu(hypotheses, references)

        result["comet"] = round(comet, 4)
        result["chrf"] = round(chrf, 2)
        result["bleu"] = round(bleu, 2)

        print("  {} quality: COMET={:.4f} chrF++={:.2f} BLEU={:.2f}".format(
            label, comet, chrf, bleu))

    # Free GPU
    del llm
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark speed with vLLM")
    parser.add_argument("--experiment", nargs="*", default=None)
    parser.add_argument("--skip-gptq", action="store_true", help="Only benchmark fp16")
    args = parser.parse_args()

    load_env()

    if args.experiment:
        exp_ids = args.experiment
    else:
        exp_ids = sorted(
            p.parent.name for p in RESULTS_DIR.glob("*/results.json")
        )

    for eid in exp_ids:
        exp_dir = RESULTS_DIR / eid
        if not (exp_dir / "results.json").exists():
            print("SKIP {}: no results".format(eid))
            continue

        fp16_path = get_fp16_model_path(exp_dir)
        if not fp16_path:
            print("SKIP {}: no fp16 model found".format(eid))
            continue

        prompts, sources, references = build_prompts(exp_dir)

        print("\n=== {} (fp16 model: {}) ===".format(eid, fp16_path[:60]))

        with open(exp_dir / "results.json") as f:
            d = json.load(f)

        # vLLM fp16 benchmark (speed + quality)
        fp16_speed = benchmark_vllm(fp16_path, prompts, label="vLLM fp16",
                                     evaluate_quality=True, references=references, sources=sources)
        d.setdefault("speed_benchmark", {})["vllm_fp16"] = fp16_speed

        # GPTQ export + benchmark (both 8-bit and 4-bit, speed + quality)
        if not args.skip_gptq:
            for bits in [8, 4]:
                try:
                    gptq_path = export_gptq(fp16_path, exp_dir, bits=bits)
                    gptq_speed = benchmark_vllm(gptq_path, prompts, label="vLLM GPTQ-{}bit".format(bits),
                                                 evaluate_quality=True, references=references, sources=sources)
                    d["speed_benchmark"]["vllm_gptq{}".format(bits)] = gptq_speed
                except Exception as e:
                    print("  GPTQ {}bit failed: {}".format(bits, e))

        # Save
        with open(exp_dir / "results.json", "w") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
