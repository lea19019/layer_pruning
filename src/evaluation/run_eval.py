"""CLI script to evaluate a model on the test set."""

import argparse
import json
from pathlib import Path

from src.config import (
    FILTERED_DIR,
    RESULTS_DIR,
    SRC_LANG_NAME,
    TGT_LANG_NAME,
    TRANSLATION_PROMPT,
)
from src.evaluation.metrics import evaluate_all
from src.evaluation.translate import translate_batch, translate_with_vllm
from src.utils import load_env


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--test-src", type=Path, default=FILTERED_DIR / "test.cs")
    parser.add_argument("--test-tgt", type=Path, default=FILTERED_DIR / "test.de")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: results/<model_name>.json)")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for inference")
    parser.add_argument("--tp-size", type=int, default=1, help="vLLM tensor parallelism")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--experiment-id", type=str, default=None,
                        help="Experiment ID (e.g., B0, M1, I1)")
    args = parser.parse_args()

    load_env()

    # Load test data
    with open(args.test_src) as f:
        sources = f.read().splitlines()
    with open(args.test_tgt) as f:
        references = f.read().splitlines()

    # Create prompts
    prompts = [
        TRANSLATION_PROMPT.format(
            src_lang=SRC_LANG_NAME,
            tgt_lang=TGT_LANG_NAME,
            source=src,
        )
        for src in sources
    ]

    # Generate translations
    if args.use_vllm:
        hypotheses = translate_with_vllm(
            args.model, prompts,
            max_new_tokens=args.max_new_tokens,
            tensor_parallel_size=args.tp_size,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="auto"
        )
        hypotheses = translate_batch(
            model, tokenizer, prompts,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )

    # Evaluate
    metrics = evaluate_all(hypotheses, references, sources)

    # Save results
    if args.output is None:
        model_name = Path(args.model).name or args.model.replace("/", "_")
        args.output = RESULTS_DIR / f"{model_name}.json"

    args.output.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "experiment_id": args.experiment_id,
        "model": args.model,
        "metrics": metrics,
        "n_samples": len(sources),
    }

    # Add translations for inspection
    result["translations"] = [
        {"source": s, "reference": r, "hypothesis": h}
        for s, r, h in zip(sources[:10], references[:10], hypotheses[:10])
    ]

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
