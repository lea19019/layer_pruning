"""Quick validation: translate a small sample and print results.

Verifies that stop strings prevent hallucination after translation.
Run on a GPU node:
    srun --partition=m13h --qos=gpu --account=sdrich --gres=gpu:h200:1 \
        --mem=64G --time=00:15:00 \
        bash -c "cd ~/attention_lp && source .venv/bin/activate && \
        export HF_HUB_OFFLINE=1 && python scripts/validate_translation.py"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    BASE_MODEL,
    FILTERED_DIR,
    SRC_LANG_NAME,
    TGT_LANG_NAME,
    TRANSLATION_PROMPT,
)
from src.evaluation.translate import translate_batch
from src.utils import load_env

N_SAMPLES = 50


def main():
    load_env()

    print(f"Loading {BASE_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto",
    )
    print(f"Loaded: {model.config.num_hidden_layers} layers on {model.device}\n")

    # Load test data
    sources = (FILTERED_DIR / "test.cs").read_text().splitlines()[:N_SAMPLES]
    references = (FILTERED_DIR / "test.de").read_text().splitlines()[:N_SAMPLES]

    prompts = [
        TRANSLATION_PROMPT.format(
            src_lang=SRC_LANG_NAME, tgt_lang=TGT_LANG_NAME, source=src,
        )
        for src in sources
    ]

    print(f"Translating {len(prompts)} sentences ...\n")
    hypotheses = translate_batch(model, tokenizer, prompts, max_new_tokens=256, batch_size=4)

    # Print results
    print("=" * 80)
    for i, (src, ref, hyp) in enumerate(zip(sources, references, hypotheses)):
        print(f"\n--- Sample {i+1} ---")
        print(f"SRC: {src[:120]}{'...' if len(src) > 120 else ''}")
        print(f"REF: {ref[:120]}{'...' if len(ref) > 120 else ''}")
        print(f"HYP: {hyp[:120]}{'...' if len(hyp) > 120 else ''}")
        # Flag if output looks suspiciously long or contains hallucination markers
        if len(hyp) > len(src) * 3:
            print("  ⚠️  OUTPUT SUSPICIOUSLY LONG — possible hallucination leak")
        if "Czech:" in hyp or "Translation:" in hyp:
            print("  ⚠️  HALLUCINATION DETECTED in output")

    print("\n" + "=" * 80)
    avg_len_ratio = sum(len(h) for h in hypotheses) / max(sum(len(r) for r in references), 1)
    print(f"\nAvg hypothesis/reference length ratio: {avg_len_ratio:.2f}")
    print("(Should be close to 1.0 — much higher suggests hallucination)")
    print("Done.")


if __name__ == "__main__":
    main()
