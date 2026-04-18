"""Push the three released fp16 models to HuggingFace under adrianMT56/.

Requires `huggingface-cli login` (or `HF_TOKEN` in the environment) before running.

Usage:

  # Push all three models
  python scripts/push_to_hf.py

  # Push a single model
  python scripts/push_to_hf.py --only I2_8_enes

  # Dry-run (prints what would happen without uploading)
  python scripts/push_to_hf.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_USER = "adrianMT56"

MODELS = {
    "B4_enes": {
        "hf_repo": "aya-enes-B4",
        "local_path": "experiments/results/B4_enes/finetuned/merged",
        "layers_removed": [],
        "n_layers": 32,
        "recipe": "Baseline: full 32-layer Aya-Expanse 8B, LoRA fine-tuning + "
                  "knowledge distillation from Aya-Expanse 32B.",
    },
    "I2_8_enes": {
        "hf_repo": "aya-enes-I2-8",
        "local_path": "experiments/results/I2_8_enes/finetuned/merged",
        "layers_removed": [8, 10, 11, 12, 13, 14, 15, 16],
        "n_layers": 24,
        "recipe": "IFR-guided layer pruning (8 middle layers removed), LoRA "
                  "fine-tuning + knowledge distillation from Aya-Expanse 32B.",
    },
    "I5_t05_kd_enes": {
        "hf_repo": "aya-enes-I5-t05-kd",
        "local_path": "experiments/results/I5_t05_kd_enes/finetuned/merged",
        "layers_removed": [8, 10, 11, 12, 13, 14, 15, 16, 17],
        "n_layers": 23,
        "recipe": "IFR threshold pruning (threshold=0.5 -> 9 layers removed), "
                  "LoRA fine-tuning + knowledge distillation from Aya-Expanse 32B.",
    },
}

MODEL_CARD_TEMPLATE = """---
language: [en, es]
license: cc-by-nc-4.0
base_model: CohereForAI/aya-expanse-8b
tags:
- translation
- machine-translation
- aya-expanse
- layer-pruning
- interpretability
pipeline_tag: translation
---

# {repo_name}

English -> Spanish translation model derived from
[CohereForAI/aya-expanse-8b](https://huggingface.co/CohereForAI/aya-expanse-8b)
(32 layers, 8B parameters).

## Recipe

{recipe}

- Number of transformer layers: **{n_layers}** (of the original 32)
- Layers removed: `{layers_removed}`
- Pruning method: **{method}**
- Fine-tuning: LoRA (r=16, alpha=32), 3 epochs on News Commentary v18 en-es
- Distillation: synthetic translations from Aya-Expanse 32B, filtered to COMET >= 0.7
- Precision: fp16

## Evaluation

Evaluated on 500 held-out News Commentary v18 en-es sentences.

| Metric | Value |
|--------|------:|
| COMET (wmt22-comet-da) | {comet:.4f} |
| chrF++ | {chrf:.2f} |
| BLEU | {bleu:.2f} |

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "{hf_user}/{repo_name}"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16)

prompt = ("Translate the following English text to Spanish.\\n\\n"
          "English: The quick brown fox jumps over the lazy dog.\\n"
          "Spanish:")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

CPU users can omit `dtype=torch.float16` (defaults to float32) or leave it as fp16
at the cost of some throughput.  For GPTQ 4-bit conversion see the project's
`scripts/quantize_to_gptq.py`.

## Reproducibility

This checkpoint was produced by the pipeline at
<https://github.com/{hf_user}/attention_lp>.
See `README.md` in that repo for the full training recipe and evaluation scripts.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", choices=sorted(MODELS), default=None,
                   help="Upload only this model (default: all three).")
    p.add_argument("--private", action="store_true",
                   help="Create the repo as private (default: public).")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't upload; just print what would happen.")
    return p.parse_args()


def load_metrics(model_key: str) -> dict[str, float]:
    results = json.loads(
        (PROJECT_ROOT / "experiments" / "results" / model_key / "results.json").read_text()
    )
    return results["metrics"]


def patch_config_dtype(local_path: Path) -> None:
    """Ensure config.json has torch_dtype set to float16 so users load fp16 by default."""
    cfg_path = local_path / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("torch_dtype") != "float16":
        cfg["torch_dtype"] = "float16"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"  Patched torch_dtype=float16 in {cfg_path}", file=sys.stderr)


def write_model_card(local_path: Path, model_key: str, info: dict) -> Path:
    metrics = load_metrics(model_key)
    method = "IFR (Information Flow Routes)" if info["layers_removed"] else "none"
    card = MODEL_CARD_TEMPLATE.format(
        hf_user=HF_USER,
        repo_name=info["hf_repo"],
        recipe=info["recipe"],
        n_layers=info["n_layers"],
        layers_removed=info["layers_removed"] or "none",
        method=method,
        comet=metrics["comet"],
        chrf=metrics["chrf"],
        bleu=metrics["bleu"],
    )
    card_path = local_path / "README.md"
    card_path.write_text(card)
    return card_path


def upload_one(model_key: str, args: argparse.Namespace) -> None:
    info = MODELS[model_key]
    local_path = PROJECT_ROOT / info["local_path"]
    if not local_path.exists():
        raise SystemExit(f"Missing local path for {model_key}: {local_path}")

    patch_config_dtype(local_path)
    card_path = write_model_card(local_path, model_key, info)
    repo_id = f"{HF_USER}/{info['hf_repo']}"

    print(f"\n=== {model_key} -> {repo_id} ===", file=sys.stderr)
    print(f"  local_path:    {local_path}", file=sys.stderr)
    print(f"  model_card:    {card_path}", file=sys.stderr)
    print(f"  layers:        {info['n_layers']} (removed: {info['layers_removed'] or 'none'})",
          file=sys.stderr)

    if args.dry_run:
        print("  [dry-run] skipping upload", file=sys.stderr)
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model",
                    private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["*.bin", "optimizer.pt", "scheduler.pt", "trainer_state.json",
                         "training_args.bin", "checkpoint-*", "*.lock"],
    )
    print(f"  uploaded: https://huggingface.co/{repo_id}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    keys = [args.only] if args.only else list(MODELS)
    for key in keys:
        upload_one(key, args)


if __name__ == "__main__":
    main()
