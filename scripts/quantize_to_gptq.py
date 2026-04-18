"""Convert a fp16 Aya-Expanse model (HF repo or local path) to GPTQ 4-bit.

The script uses `gptqmodel` for quantization.  Calibration text defaults to
128 lines from `data/filtered_en_es/test.en` (falling back to `data/filtered/test.cs`)
but you can pass your own file with `--calibration`.

Usage:

  # Quantize a released model and save locally
  python scripts/quantize_to_gptq.py \
      --model adrianMT56/aya-enes-I5-t05-kd \
      --output-dir gptq_out/I5_t05_kd_enes

  # Custom calibration text
  python scripts/quantize_to_gptq.py --model <id> --output-dir out/ \
      --calibration my_calib.txt --num-samples 256

Notes:
 * Quantization requires a GPU. Running on CPU is not supported by gptqmodel.
 * ~16 GB of GPU VRAM is enough for 8B-class models; the pruned checkpoints
   (24 or 22 layers) fit comfortably in 12 GB.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True,
                   help="HuggingFace repo ID or local path to the fp16 model.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory where the GPTQ checkpoint will be written.")
    p.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--calibration", type=Path, default=None,
                   help="Text file with one sample per line (default: data/filtered_en_es/test.en).")
    p.add_argument("--num-samples", type=int, default=128,
                   help="How many calibration lines to use (default: 128).")
    return p.parse_args()


def load_calibration(args: argparse.Namespace) -> list[str]:
    if args.calibration is not None:
        lines = Path(args.calibration).read_text(encoding="utf-8").splitlines()
    else:
        candidates = [
            PROJECT_ROOT / "data" / "filtered_en_es" / "test.en",
            PROJECT_ROOT / "data" / "filtered" / "test.cs",
        ]
        for path in candidates:
            if path.exists():
                lines = path.read_text(encoding="utf-8").splitlines()
                print(f"Using calibration file: {path}", file=sys.stderr)
                break
        else:
            raise SystemExit(
                "No calibration file found. Pass --calibration <path> or unpack "
                "data.zip to get the default calibration set."
            )
    lines = [ln for ln in lines if ln.strip()][: args.num_samples]
    if not lines:
        raise SystemExit("Calibration file is empty after filtering.")
    return lines


def main() -> None:
    args = parse_args()

    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    calib = load_calibration(args)
    print(f"Quantizing {args.model} to INT{args.bits} with {len(calib)} samples...",
          file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    quant_cfg = QuantizeConfig(bits=args.bits, group_size=args.group_size)
    model = GPTQModel.load(args.model, quantize_config=quant_cfg)
    model.quantize(calib, tokenizer=tokenizer)
    model.save(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved GPTQ model to {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
