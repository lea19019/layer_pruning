#!/usr/bin/env python3
"""Run BnB INT8 evaluation on existing finetuned models.

For I3/M3 (no KD) experiments, loads the I1/M1 finetuned model.
For I4/M4 (with KD) experiments, loads the I2/M2 finetuned model.
Applies BnB INT8 quantization at load time and evaluates.

Usage:
    python scripts/run_bnb_eval.py M3_8_int8_enes M3_12_int8_enes ...
    python scripts/run_bnb_eval.py --all-missing
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TRANSLATION_PROMPT
from src.evaluation.metrics import compute_comet, compute_chrf, compute_bleu
from src.evaluation.translate import translate_batch
from src.run_experiment import _resolve_lang_pair, _data_dir

RESULTS_DIR = Path("experiments/results")

# Map BnB experiment -> fp16 equivalent
EQUIV_MAP = {
    # I3 (IFR + FT + BnB) -> I1 (IFR + FT)
    "I3": "I1", "I4": "I2",
    # M3 (Heuristic + FT + BnB) -> M1 (Heuristic + FT)
    "M3": "M1", "M4": "M2",
}


def get_fp16_equiv(exp_id: str) -> str:
    """Map a BnB experiment to its fp16 equivalent."""
    for bnb_prefix, fp16_prefix in EQUIV_MAP.items():
        if exp_id.startswith(bnb_prefix):
            suffix = exp_id[len(bnb_prefix):]
            # Remove _int8 from suffix to get base experiment
            suffix = suffix.replace("_int8", "")
            return fp16_prefix + suffix
    return None


def run_bnb_eval(exp_id: str):
    config_path = Path("experiments/configs/{}.yaml".format(exp_id))
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_dir = RESULTS_DIR / exp_id
    results_file = exp_dir / "results.json"
    if results_file.exists():
        print("SKIP {}: results already exist".format(exp_id))
        return

    # Find the fp16 model
    equiv = get_fp16_equiv(exp_id)
    if not equiv:
        print("ERROR {}: can't find fp16 equivalent".format(exp_id))
        return

    fp16_model_path = RESULTS_DIR / equiv / "finetuned" / "merged"
    if not (fp16_model_path / "config.json").exists():
        print("ERROR {}: fp16 model not found at {}".format(exp_id, fp16_model_path))
        return

    print("\n=== {} ===".format(exp_id))
    print("  Loading {} with BnB INT8".format(fp16_model_path))

    # Load with BnB INT8
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_bits = cfg.get("quantization", {}).get("bits", 8)
    if quant_bits == 8:
        qconfig = BitsAndBytesConfig(load_in_8bit=True)
    else:
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=cfg.get("quantization", {}).get("quant_type", "nf4"),
        )

    tokenizer = AutoTokenizer.from_pretrained(str(fp16_model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(fp16_model_path),
        quantization_config=qconfig,
        device_map="auto",
    )

    # Load test data
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

    # Translate and evaluate
    hypotheses = translate_batch(model, tokenizer, prompts)
    comet = compute_comet(hypotheses, references, sources)
    chrf = compute_chrf(hypotheses, references)
    bleu = compute_bleu(hypotheses, references)

    # Model size
    model_size_mb = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    ) / (1024 * 1024)

    # Inference speed (quick estimate)
    import time
    speed_prompts = prompts[:20]
    start = time.perf_counter()
    speed_hyps = translate_batch(model, tokenizer, speed_prompts)
    elapsed = time.perf_counter() - start
    total_tokens = sum(len(tokenizer.encode(h)) for h in speed_hyps)
    tps = total_tokens / elapsed

    print("  COMET={:.4f} chrF++={:.2f} BLEU={:.2f}".format(comet, chrf, bleu))
    print("  Size={:.0f} MB, Speed={:.1f} tok/s".format(model_size_mb, tps))

    # Save results
    exp_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "config": cfg,
        "metrics": {
            "comet": comet,
            "chrf": chrf,
            "bleu": bleu,
            "model_size": {"size_mb": model_size_mb},
            "inference_speed": {"tokens_per_second": tps, "n_samples": 20},
        },
        "fp16_source": equiv,
    }
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Free GPU
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="*")
    parser.add_argument("--all-missing", action="store_true")
    args = parser.parse_args()

    if args.all_missing:
        exp_ids = []
        for p in sorted(Path("experiments/configs").glob("*.yaml")):
            name = p.stem
            # Only BnB experiments
            if not any(name.startswith(prefix) for prefix in ["I3_", "I4_", "M3_", "M4_"]):
                continue
            if (RESULTS_DIR / name / "results.json").exists():
                continue
            exp_ids.append(name)
    else:
        exp_ids = args.experiments

    print("Running BnB eval for: {}".format(exp_ids))
    for exp_id in exp_ids:
        try:
            run_bnb_eval(exp_id)
        except Exception as e:
            print("ERROR {}: {}".format(exp_id, e))


if __name__ == "__main__":
    main()
