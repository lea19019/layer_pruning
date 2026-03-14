"""INT4 quantization using BitsAndBytes."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils import load_env


def quantize_model(
    model_name_or_path: str,
    output_dir: Path,
    quant_type: str = "nf4",
    compute_dtype: torch.dtype = torch.float16,
) -> Path:
    """Load a model with INT4 quantization and save it.

    Args:
        model_name_or_path: HuggingFace model or path.
        output_dir: Where to save quantized model.
        quant_type: Quantization type (nf4 or fp4).
        compute_dtype: Compute dtype for quantized layers.

    Returns:
        Path to saved quantized model.
    """
    load_env()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {model_name_or_path} with INT4 quantization ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print(f"Saving quantized model to {output_dir} ...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Report size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Quantized model saved to {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Quantize model to INT4")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quant-type", default="nf4", choices=["nf4", "fp4"])
    args = parser.parse_args()

    quantize_model(args.model, args.output_dir, args.quant_type)


if __name__ == "__main__":
    main()
