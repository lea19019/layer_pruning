#!/usr/bin/env python3
"""Interactive CLI demo: load one translation model and translate sentences.

Run in separate terminals with different models to compare side-by-side.

Usage:
    python scripts/demo.py B4_enes           # load unpruned baseline
    python scripts/demo.py I2_8_enes         # load IFR-pruned (24 layers)
    python scripts/demo.py I2_16_enes        # load IFR-pruned (16 layers, fastest)

Commands in the prompt:
    <type any sentence>  -> translate it
    :N                   -> translate example sentence N (see :list)
    :list                -> show example sentences
    :all                 -> translate all example sentences
    :q                   -> quit
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Force CPU to ensure consistent timing comparisons across terminals
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATHS = {
    # English -> Spanish models
    "B4_enes": "experiments/results/B4_enes/finetuned/merged",
    "I2_8_enes": "experiments/results/I2_8_enes/finetuned/merged",
    "I2_12_enes": "experiments/results/I2_12_enes/finetuned/merged",
    "I2_16_enes": "experiments/results/I2_16_enes/finetuned/merged",
    "M2_8_enes": "experiments/results/M2_8_enes/finetuned/merged",
    "M2_12_enes": "experiments/results/M2_12_enes/finetuned/merged",
    "I5_t05_kd_enes": "experiments/results/I5_t05_kd_enes/finetuned/merged",
    "IP_16_enes": "experiments/results/IP_16_enes/pruned_model",
}

EXAMPLE_SENTENCES = [
    # News-style sentences matching the training distribution (News Commentary v18)
    "Indeed, one hopes that others will follow his example.",
    "Today, the program remains the only tangible result of the investment.",
    "The central bank announced a significant reduction in interest rates this morning.",
    "Scientists discovered that climate change is accelerating faster than previously estimated.",
    "The government has promised major reforms to address the ongoing economic crisis.",
    "Researchers at the university have published groundbreaking findings on renewable energy.",
    "International negotiations continue, but no agreement has been reached on the trade dispute.",
    "Only by focusing on these fundamentals can we build a more sustainable future.",
]

PROMPT_TEMPLATE = "Translate the following English sentence to Spanish: {source}"


def load_model(model_name: str):
    if model_name not in MODEL_PATHS:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(MODEL_PATHS.keys())}")
        sys.exit(1)

    path = MODEL_PATHS[model_name]
    if not Path(path).exists():
        print(f"Model path does not exist: {path}")
        sys.exit(1)

    # Check disk size before loading
    model_dir = Path(path)
    disk_bytes = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    disk_gb = disk_bytes / (1024 ** 3)

    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")
    print(f"  Path:      {path}")
    print(f"  Disk size: {disk_gb:.2f} GB")
    print(f"  Device:    CPU (float32)")
    print()
    print(f"Loading tokenizer...", flush=True)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(path)
    print(f"  OK ({time.perf_counter() - t0:.1f}s)")

    print(f"Loading model weights (this takes ~20-60s)...", flush=True)
    t1 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    weights_time = time.perf_counter() - t1
    print(f"  OK ({weights_time:.1f}s)")

    # Compute stats
    total_params = sum(p.numel() for p in model.parameters())
    ram_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    load_time = time.perf_counter() - t0

    print()
    print(f"{'='*60}")
    print(f"READY: {model_name}")
    print(f"{'='*60}")
    print(f"  Layers:      {layers}  ({32 - layers} removed from 32)")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Parameters:  {total_params / 1e9:.2f} B ({total_params:,})")
    print(f"  RAM usage:   {ram_mb / 1024:.2f} GB (fp32)")
    print(f"  Load time:   {load_time:.1f}s")
    print(f"{'='*60}\n")

    return model, tokenizer, layers


def translate(model, tokenizer, source: str, max_new_tokens: int = 200) -> tuple[str, float]:
    # Use chat template — Aya Expanse is instruction-tuned and expects chat format.
    # Raw text prompting causes the model to generate EOS immediately.
    prompt = PROMPT_TEMPLATE.format(source=source)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    input_len = input_ids.shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    elapsed = time.perf_counter() - t0

    generated_tokens = outputs[0][input_len:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Strip common trailing patterns (model sometimes keeps generating examples)
    for stop in ["\n\nTranslate", "\nEnglish:", "\nSpanish:", "\n\n"]:
        if stop in text:
            text = text.split(stop)[0].strip()
            break

    tokens_generated = len(generated_tokens)
    tps = tokens_generated / elapsed if elapsed > 0 else 0

    return text.strip(), elapsed, tokens_generated, tps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Model name (e.g., B4_enes, I2_8_enes, I2_16_enes)")
    args = parser.parse_args()

    model, tokenizer, layers = load_model(args.model)

    print("Commands: :list (show examples), :N (translate example N), :all, :q (quit)")
    print("Or type any English sentence to translate.\n")

    while True:
        try:
            user_input = input("EN> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input in [":q", ":quit", ":exit"]:
            print("Bye!")
            break

        if user_input == ":list":
            print("\nExample sentences:")
            for i, s in enumerate(EXAMPLE_SENTENCES, 1):
                print(f"  {i}. {s}")
            print()
            continue

        if user_input == ":all":
            print("\nTranslating all examples...\n")
            total_time = 0
            total_tokens = 0
            for i, sentence in enumerate(EXAMPLE_SENTENCES, 1):
                print(f"[{i}] EN: {sentence}")
                translation, elapsed, tokens, tps = translate(model, tokenizer, sentence)
                print(f"    ES: {translation}")
                print(f"    {elapsed:.2f}s  |  {tokens} tokens  |  {tps:.1f} tok/s\n")
                total_time += elapsed
                total_tokens += tokens
            avg_tps = total_tokens / total_time if total_time > 0 else 0
            print(f"--- Summary ({args.model}, {layers} layers) ---")
            print(f"Total: {total_time:.2f}s  |  {total_tokens} tokens  |  avg {avg_tps:.1f} tok/s\n")
            continue

        # :N shortcut for example N
        if user_input.startswith(":") and user_input[1:].isdigit():
            idx = int(user_input[1:]) - 1
            if 0 <= idx < len(EXAMPLE_SENTENCES):
                sentence = EXAMPLE_SENTENCES[idx]
                print(f"EN: {sentence}")
            else:
                print(f"Invalid example number. Use :list to see options.")
                continue
        else:
            sentence = user_input

        translation, elapsed, tokens, tps = translate(model, tokenizer, sentence)
        print(f"ES: {translation}")
        print(f"    {elapsed:.2f}s  |  {tokens} tokens  |  {tps:.1f} tok/s\n")


if __name__ == "__main__":
    main()
