"""Experiment 5/6/7: Fine-tuning recovery curve.

Retrains the pruned model (IP_16_enes) with configurable data and evaluates
at intermediate checkpoints to track how COMET recovers during training.

Supports three ablation axes:
  - Recovery curve: full data (authentic + KD), evaluate at many checkpoints
  - No-KD ablation: authentic data only (--no-kd flag)
  - Data fraction ablation: subset of data (--data-fraction 0.25/0.5/0.75)

Saves only LoRA adapter weights at each checkpoint (~80MB each instead of
8.5GB for a merged model). During evaluation, adapters are merged in memory
— no extra disk usage.

Usage:
    # Full recovery curve with KD data (default: checkpoint every 1000 steps)
    python -m ablation.scripts.ft_recovery_curve

    # Without KD
    python -m ablation.scripts.ft_recovery_curve --no-kd

    # Data fraction ablation
    python -m ablation.scripts.ft_recovery_curve --data-fraction 0.25

    # Evaluate existing checkpoints only (no training)
    python -m ablation.scripts.ft_recovery_curve --eval-only ablation/results/ft_recovery/full_kd
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from src.config import (
    FT_BATCH_SIZE,
    FT_EPOCHS,
    FT_GRAD_ACCUM,
    FT_LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
)
from src.utils import set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "ablation/results"

PRUNED_MODEL = str(PROJECT_ROOT / "experiments/results/IP_16_enes/pruned_model")
TRAIN_EN = PROJECT_ROOT / "data/filtered_en_es/train.en"
TRAIN_ES = PROJECT_ROOT / "data/filtered_en_es/train.es"
KD_EN = PROJECT_ROOT / "data/kd_en_es/kd.en"
KD_ES = PROJECT_ROOT / "data/kd_en_es/kd.es"
TEST_EN = PROJECT_ROOT / "data/filtered_en_es/test.en"
TEST_ES = PROJECT_ROOT / "data/filtered_en_es/test.es"

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


class AdapterCheckpointCallback(TrainerCallback):
    """Save LoRA adapter weights at specified step intervals.

    Only saves adapter weights (~80MB), not the full merged model (~8.5GB).
    """

    def __init__(self, save_steps: int, output_dir: Path, tokenizer):
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.saved_steps = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.save_steps == 0 and state.global_step > 0:
            self._save(model, state.global_step)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step not in self.saved_steps:
            self._save(model, step, tag=f"epoch_{state.epoch:.0f}")

    def _save(self, model, step, tag=None):
        name = tag or f"step_{step}"
        save_dir = self.output_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> Saving adapter: {name} (step {step})")
        model.save_pretrained(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        self.saved_steps.append(step)


def load_parallel_data(
    src_path: Path, tgt_path: Path, fraction: float = 1.0
) -> list[tuple[str, str]]:
    """Load parallel text files, optionally subsampled."""
    with open(src_path) as f:
        sources = f.read().splitlines()
    with open(tgt_path) as f:
        targets = f.read().splitlines()
    pairs = list(zip(sources, targets))
    if fraction < 1.0:
        n = max(1, int(len(pairs) * fraction))
        random.shuffle(pairs)
        pairs = pairs[:n]
    return pairs


def build_dataset(pairs: list[tuple[str, str]]) -> Dataset:
    """Convert (source, target) pairs to training dataset."""
    texts = []
    for src, tgt in pairs:
        prompt = f"Translate the following English sentence to Spanish: {src}"
        texts.append(prompt + " " + tgt)
    return Dataset.from_dict({"text": texts})


def evaluate_adapter_in_memory(
    adapter_dir: str,
    base_model_path: str,
    sources: list[str],
    references: list[str],
    device: str = "cuda",
    max_samples: int = 200,
) -> dict:
    """Load adapter, merge in memory, evaluate, discard. No disk writes."""
    from src.evaluation.metrics import compute_bleu, compute_chrf, compute_comet
    from src.evaluation.translate import translate_batch

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.merge_and_unload()
    model.eval()

    eval_sources = sources[:max_samples]
    eval_references = references[:max_samples]
    prompts = [
        f"Translate the following English sentence to Spanish: {s}"
        for s in eval_sources
    ]

    hypotheses = translate_batch(model, tokenizer, prompts, batch_size=8)

    comet = compute_comet(hypotheses, eval_references, eval_sources)
    chrf = compute_chrf(hypotheses, eval_references)
    bleu = compute_bleu(hypotheses, eval_references)

    del model, base_model
    torch.cuda.empty_cache()

    return {"comet": comet, "chrf": chrf, "bleu": bleu}


def evaluate_merged_model(
    model_path: str,
    sources: list[str],
    references: list[str],
    device: str = "cuda",
    max_samples: int = 200,
) -> dict:
    """Evaluate an already-merged model."""
    from src.evaluation.metrics import compute_bleu, compute_chrf, compute_comet
    from src.evaluation.translate import translate_batch

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    eval_sources = sources[:max_samples]
    eval_references = references[:max_samples]
    prompts = [
        f"Translate the following English sentence to Spanish: {s}"
        for s in eval_sources
    ]

    hypotheses = translate_batch(model, tokenizer, prompts, batch_size=8)

    comet = compute_comet(hypotheses, eval_references, eval_sources)
    chrf = compute_chrf(hypotheses, eval_references)
    bleu = compute_bleu(hypotheses, eval_references)

    del model
    torch.cuda.empty_cache()

    return {"comet": comet, "chrf": chrf, "bleu": bleu}


def train_with_checkpoints(
    use_kd: bool = True,
    data_fraction: float = 1.0,
    save_steps: int = 1000,
    epochs: int = FT_EPOCHS,
    device: str = "cuda",
    run_name: str = "recovery_curve",
):
    """Train pruned model, saving adapter-only checkpoints."""
    set_seed(42)

    output_dir = RESULTS_DIR / "ft_recovery" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading training data...")
    pairs = load_parallel_data(TRAIN_EN, TRAIN_ES, fraction=data_fraction)
    print(f"  Authentic: {len(pairs)} pairs (fraction={data_fraction})")

    if use_kd and KD_EN.exists() and KD_ES.exists():
        kd_pairs = load_parallel_data(KD_EN, KD_ES, fraction=data_fraction)
        pairs.extend(kd_pairs)
        print(f"  + KD: {len(kd_pairs)} pairs -> Total: {len(pairs)}")

    dataset = build_dataset(pairs)

    # Load model
    print(f"Loading pruned model from {PRUNED_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(PRUNED_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        PRUNED_MODEL, torch_dtype=torch.bfloat16, device_map=device
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Calculate total steps for reference
    n_batches = len(dataset) // FT_BATCH_SIZE
    steps_per_epoch = n_batches // FT_GRAD_ACCUM
    total_steps = steps_per_epoch * epochs
    print(f"Estimated: {steps_per_epoch} steps/epoch, {total_steps} total steps")

    # Disable HF Trainer's own checkpointing (we use our callback)
    training_args = SFTConfig(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=FT_BATCH_SIZE,
        gradient_accumulation_steps=FT_GRAD_ACCUM,
        learning_rate=FT_LEARNING_RATE,
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        max_length=512,
    )

    checkpoints_dir = output_dir / "checkpoints"
    callback = AdapterCheckpointCallback(save_steps, checkpoints_dir, tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[callback],
    )

    print(f"\nStarting training: {epochs} epochs, save every {save_steps} steps")
    trainer.train()

    # Save final adapter
    final_dir = checkpoints_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Final adapter saved to {final_dir}")

    # Clean up HF Trainer's output dir (logs, etc.)
    trainer_dir = output_dir / "trainer"
    if trainer_dir.exists():
        shutil.rmtree(trainer_dir, ignore_errors=True)

    # Save training config
    config = {
        "use_kd": use_kd,
        "data_fraction": data_fraction,
        "save_steps": save_steps,
        "epochs": epochs,
        "n_training_examples": len(dataset),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "saved_checkpoints": [c.name for c in sorted(checkpoints_dir.iterdir())],
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    return output_dir


def evaluate_all_checkpoints(
    run_dir: Path, device: str = "cuda", max_eval_samples: int = 200
):
    """Evaluate all saved adapter checkpoints by merging in memory."""
    print("\n=== Evaluating checkpoints ===")

    with open(TEST_EN) as f:
        sources = [line.strip() for line in f if line.strip()]
    with open(TEST_ES) as f:
        references = [line.strip() for line in f if line.strip()]

    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        print(f"No checkpoints dir at {checkpoints_dir}")
        return []

    results = []

    # Sort checkpoint dirs: step_1000 < step_2000 < ... < epoch_1 < epoch_2 < final
    def sort_key(p):
        name = p.name
        if name.startswith("step_"):
            return (0, int(name.split("_")[1]))
        elif name.startswith("epoch_"):
            return (1, int(name.split("_")[1]))
        elif name == "final":
            return (2, 0)
        return (3, 0)

    checkpoint_dirs = sorted(
        [d for d in checkpoints_dir.iterdir() if d.is_dir()],
        key=sort_key,
    )

    for ckpt_dir in checkpoint_dirs:
        name = ckpt_dir.name
        # Check if this is an adapter checkpoint (has adapter_config.json)
        is_adapter = (ckpt_dir / "adapter_config.json").exists()
        # Or a merged model (has config.json but no adapter_config.json)
        is_merged = (ckpt_dir / "config.json").exists() and not is_adapter
        # Could also be in a "merged" subdir (old format)
        merged_subdir = ckpt_dir / "merged"
        has_merged_subdir = merged_subdir.exists() and (merged_subdir / "config.json").exists()

        print(f"\nEvaluating: {name}")
        try:
            if is_adapter:
                scores = evaluate_adapter_in_memory(
                    str(ckpt_dir), PRUNED_MODEL,
                    sources, references, device, max_eval_samples,
                )
            elif is_merged:
                scores = evaluate_merged_model(
                    str(ckpt_dir), sources, references, device, max_eval_samples,
                )
            elif has_merged_subdir:
                scores = evaluate_merged_model(
                    str(merged_subdir), sources, references, device, max_eval_samples,
                )
            else:
                print(f"  Skipping: no adapter or model found")
                continue

            scores["checkpoint"] = name
            results.append(scores)
            print(f"  COMET={scores['comet']:.4f}, chrF={scores['chrf']:.2f}, "
                  f"BLEU={scores['bleu']:.2f}")
        except Exception as e:
            print(f"  FAILED: {e}")

    # Save evaluation results
    with open(run_dir / "recovery_curve.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRecovery curve saved to {run_dir}/recovery_curve.json")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-kd", action="store_true",
                        help="Train without KD data (authentic only)")
    parser.add_argument("--data-fraction", type=float, default=1.0,
                        help="Fraction of data to use (0.0-1.0)")
    parser.add_argument("--save-steps", type=int, default=1000,
                        help="Save adapter checkpoint every N steps")
    parser.add_argument("--epochs", type=int, default=FT_EPOCHS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-only", type=Path, default=None,
                        help="Skip training, just evaluate checkpoints in this dir")
    parser.add_argument("--max-eval-samples", type=int, default=200,
                        help="Max test samples per checkpoint evaluation")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_all_checkpoints(args.eval_only, args.device, args.max_eval_samples)
        return

    # Determine run name
    parts = []
    if args.no_kd:
        parts.append("no_kd")
    if args.data_fraction < 1.0:
        parts.append(f"frac_{args.data_fraction}")
    run_name = "_".join(parts) if parts else "full_kd"

    run_dir = train_with_checkpoints(
        use_kd=not args.no_kd,
        data_fraction=args.data_fraction,
        save_steps=args.save_steps,
        epochs=args.epochs,
        device=args.device,
        run_name=run_name,
    )

    evaluate_all_checkpoints(run_dir, args.device, args.max_eval_samples)


if __name__ == "__main__":
    main()
