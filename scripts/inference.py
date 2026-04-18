"""CPU-friendly inference for the released pruned Aya-Expanse models.

Usage examples:

  # Translate a single line from the CLI (en -> es by default)
  python scripts/inference.py --model adrianMT56/aya-enes-B4 \
      --text "The quick brown fox jumps over the lazy dog."

  # Translate a file (one sentence per line), output to another file
  python scripts/inference.py --model adrianMT56/aya-enes-I5-t05-kd \
      --input data/filtered_en_es/test.en --output translations.es

  # Czech -> German
  python scripts/inference.py --model adrianMT56/aya-enes-B4 \
      --src-lang Czech --tgt-lang German --text "Rychla hneda liska."

  # Use GPU if available (default is CPU)
  python scripts/inference.py --model <id> --device cuda --text "..."

The script loads the model from a local path or a HuggingFace repo ID.  Pruned
models keep their reduced `num_hidden_layers` in config.json, so no special
handling is required on the loading side.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.translate import translate_batch  # noqa: E402


PROMPT_TEMPLATE = (
    "Translate the following {src_lang} text to {tgt_lang}.\n\n"
    "{src_lang}: {source}\n{tgt_lang}:"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True,
                   help="HuggingFace repo ID or local path to a merged fp16 model.")
    p.add_argument("--text", default=None, help="One line of source text to translate.")
    p.add_argument("--input", type=Path, default=None,
                   help="Input file, one source sentence per line.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output file (defaults to stdout).")
    p.add_argument("--src-lang", default="English",
                   help="Source language name used in the prompt (default: English).")
    p.add_argument("--tgt-lang", default="Spanish",
                   help="Target language name used in the prompt (default: Spanish).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                   help="Device to run on (default: cpu).")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"],
                   help="Compute dtype. CPU users should keep float32 for correctness; "
                        "float16 is faster on GPU (default: float32).")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Batch size for generation (default: 4).")
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def read_sources(args: argparse.Namespace) -> list[str]:
    if args.text is not None:
        return [args.text]
    if args.input is not None:
        return Path(args.input).read_text(encoding="utf-8").splitlines()
    # Fall back to stdin
    data = sys.stdin.read()
    if not data.strip():
        raise SystemExit("No input provided. Pass --text, --input, or pipe via stdin.")
    return data.splitlines()


def main() -> None:
    args = parse_args()
    sources = read_sources(args)
    prompts = [
        PROMPT_TEMPLATE.format(src_lang=args.src_lang, tgt_lang=args.tgt_lang, source=s)
        for s in sources
    ]

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    print(f"Loading model {args.model} on {args.device} ({args.dtype})...",
          file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device_map = {"auto": "auto", "cuda": "cuda", "cpu": "cpu"}[args.device]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=device_map,
    )

    hypotheses = translate_batch(
        model, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    out = "\n".join(hypotheses) + "\n"
    if args.output is not None:
        Path(args.output).write_text(out, encoding="utf-8")
        print(f"Wrote {len(hypotheses)} translation(s) to {args.output}",
              file=sys.stderr)
    else:
        sys.stdout.write(out)


if __name__ == "__main__":
    main()
