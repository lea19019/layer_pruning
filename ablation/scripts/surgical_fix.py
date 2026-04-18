"""Surgical fixes for the pruned model's output readout.

Four approaches to test the hypothesis that pruning primarily breaks the
output readout (not internal representations), ordered cheapest to most
invasive:

  1. norm_rescale: Rescale the final RMSNorm to match the target model's
     residual magnitudes. No training.

  2. linear_probe: Fit a closed-form linear map T that maps the pruned
     model's last-layer hidden to the target model's last-layer hidden.
     No training. Pure least-squares.

  3. lm_head_ft: Freeze everything; train only lm_head + final norm.
     ~4M trainable params.

  4. mlp_last_ft: Freeze everything; train only the MLPs in the last 3
     layers. ~60M trainable params.

Usage:
    python -m ablation.scripts.surgical_fix --approach norm_rescale
    python -m ablation.scripts.surgical_fix --approach linear_probe
    python -m ablation.scripts.surgical_fix --approach lm_head_ft
    python -m ablation.scripts.surgical_fix --approach mlp_last_ft
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.utils import set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "ablation/results/surgical_fix"

PRUNED_MODEL = str(PROJECT_ROOT / "experiments/results/IP_16_enes/pruned_model")
TARGET_MODEL = str(
    PROJECT_ROOT / "experiments/results/I2_16_enes/finetuned/merged"
)  # Same 16-layer architecture — legit target for hidden-state matching

TRAIN_EN = PROJECT_ROOT / "data/filtered_en_es/train.en"
TRAIN_ES = PROJECT_ROOT / "data/filtered_en_es/train.es"
TEST_EN = PROJECT_ROOT / "data/filtered_en_es/test.en"
TEST_ES = PROJECT_ROOT / "data/filtered_en_es/test.es"


def format_prompt(src: str) -> str:
    return f"Translate the following English sentence to Spanish: {src}"


def load_model(path: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def collect_last_residuals(
    model, tokenizer, prompts: list[str], device: str = "cuda",
    n_samples: int = 500, batch_size: int = 4,
) -> np.ndarray:
    """Collect the pre-norm last-layer residual for each prompt.

    Returns:
        (n_samples, hidden_size) array — one vector per prompt (last token).
    """
    prompts = prompts[:n_samples]
    residuals = []

    # Hook on model.model.norm to capture its INPUT (the pre-norm residual)
    captured = {}

    def pre_norm_hook(module, args, kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        captured["h"] = hidden.detach()

    h = model.model.norm.register_forward_pre_hook(pre_norm_hook, with_kwargs=True)

    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            ).to(device)
            with torch.no_grad():
                model(**inputs)
            # last non-pad position
            seq_lens = inputs["attention_mask"].sum(dim=1) - 1
            hidden = captured["h"]  # (batch, seq, hidden)
            last = hidden[torch.arange(hidden.size(0)), seq_lens]
            residuals.append(last.cpu().float().numpy())
    finally:
        h.remove()

    return np.concatenate(residuals, axis=0)


def evaluate_model(model, tokenizer, sources, references, max_samples=200):
    """Evaluate a model on COMET, chrF, BLEU."""
    from src.evaluation.metrics import compute_bleu, compute_chrf, compute_comet
    from src.evaluation.translate import translate_batch

    eval_sources = sources[:max_samples]
    eval_references = references[:max_samples]
    prompts = [format_prompt(s) for s in eval_sources]

    hypotheses = translate_batch(model, tokenizer, prompts, batch_size=8)

    return {
        "comet": compute_comet(hypotheses, eval_references, eval_sources),
        "chrf": compute_chrf(hypotheses, eval_references),
        "bleu": compute_bleu(hypotheses, eval_references),
    }


# ──────────────────────────────────────────────────────────────────────────
# Approach 1: Norm recalibration (no training)
# ──────────────────────────────────────────────────────────────────────────


def compute_rms_scale(pruned_residuals: np.ndarray, target_residuals: np.ndarray) -> float:
    """Compute scalar to rescale pruned residuals to match target RMS."""
    pruned_rms = np.sqrt((pruned_residuals ** 2).mean())
    target_rms = np.sqrt((target_residuals ** 2).mean())
    return float(target_rms / (pruned_rms + 1e-8))


def approach_norm_rescale(device: str = "cuda", n_samples: int = 500,
                          max_eval_samples: int = 200):
    """Scale the final RMSNorm weight to match target model's residual magnitudes."""
    print("=== Approach 1: Norm Recalibration ===")

    with open(TEST_EN) as f:
        sources = [line.strip() for line in f if line.strip()]
    with open(TEST_ES) as f:
        references = [line.strip() for line in f if line.strip()]

    prompts = [format_prompt(s) for s in sources[:n_samples]]

    # Collect residuals from pruned model
    print("Collecting pruned-model residuals...")
    pruned_model, pruned_tok = load_model(PRUNED_MODEL, device)
    pruned_res = collect_last_residuals(pruned_model, pruned_tok, prompts, device)
    print(f"  Pruned residuals shape: {pruned_res.shape}, RMS: "
          f"{np.sqrt((pruned_res**2).mean()):.4f}")

    # Collect residuals from target model (I2_16_enes)
    print("Collecting target-model residuals...")
    del pruned_model
    torch.cuda.empty_cache()
    target_model, target_tok = load_model(TARGET_MODEL, device)
    target_res = collect_last_residuals(target_model, target_tok, prompts, device)
    print(f"  Target residuals shape: {target_res.shape}, RMS: "
          f"{np.sqrt((target_res**2).mean()):.4f}")

    scale = compute_rms_scale(pruned_res, target_res)
    print(f"  Scale factor: {scale:.4f}")

    del target_model
    torch.cuda.empty_cache()

    # Reload pruned model and modify the norm weight
    print("\nApplying scale to pruned model's final norm...")
    fixed_model, fixed_tok = load_model(PRUNED_MODEL, device)
    with torch.no_grad():
        fixed_model.model.norm.weight.data *= scale

    # Evaluate
    print("Evaluating...")
    scores = evaluate_model(fixed_model, fixed_tok, sources, references,
                            max_eval_samples)
    print(f"  COMET={scores['comet']:.4f}, chrF={scores['chrf']:.2f}, "
          f"BLEU={scores['bleu']:.2f}")

    del fixed_model
    torch.cuda.empty_cache()

    return {
        "approach": "norm_rescale",
        "scale_factor": scale,
        "pruned_rms": float(np.sqrt((pruned_res**2).mean())),
        "target_rms": float(np.sqrt((target_res**2).mean())),
        "n_collection_samples": n_samples,
        **scores,
    }


# ──────────────────────────────────────────────────────────────────────────
# Approach 2: Linear probe (closed-form)
# ──────────────────────────────────────────────────────────────────────────


def fit_linear_probe(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Fit linear map T: X @ T ≈ Y via ridge regression.

    Args:
        X: (n, d) source (pruned) residuals
        Y: (n, d) target (full-FT) residuals
        ridge: L2 regularization strength

    Returns:
        T: (d, d) linear map.
    """
    d = X.shape[1]
    XtX = X.T @ X + ridge * np.eye(d)
    XtY = X.T @ Y
    T = np.linalg.solve(XtX, XtY)
    return T


class LinearProbeHook:
    """Applies a learned linear map to the residual before the final norm."""

    def __init__(self, T: torch.Tensor):
        self.T = T

    def __call__(self, module, args, kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        if hidden is None:
            return
        # hidden: (batch, seq, hidden) — apply T along last dim
        orig_dtype = hidden.dtype
        fixed = hidden.float() @ self.T
        fixed = fixed.to(orig_dtype)
        # Return modified args so downstream gets the fixed tensor
        if args:
            new_args = (fixed,) + args[1:]
            return new_args, kwargs
        kwargs["hidden_states"] = fixed
        return args, kwargs


def approach_linear_probe(device: str = "cuda", n_samples: int = 500,
                          max_eval_samples: int = 200, ridge: float = 1e-3):
    """Fit closed-form linear map, apply via hook, evaluate."""
    print("=== Approach 2: Linear Probe ===")

    with open(TEST_EN) as f:
        sources = [line.strip() for line in f if line.strip()]
    with open(TEST_ES) as f:
        references = [line.strip() for line in f if line.strip()]

    # Use a separate pool of prompts for fitting (don't leak test data)
    with open(TRAIN_EN) as f:
        train_sources = [line.strip() for line in f if line.strip()]
    fit_prompts = [format_prompt(s) for s in train_sources[:n_samples]]

    # Collect paired residuals
    print(f"Collecting {n_samples} paired residuals...")
    pruned_model, pruned_tok = load_model(PRUNED_MODEL, device)
    pruned_res = collect_last_residuals(pruned_model, pruned_tok, fit_prompts, device)
    del pruned_model
    torch.cuda.empty_cache()

    target_model, target_tok = load_model(TARGET_MODEL, device)
    target_res = collect_last_residuals(target_model, target_tok, fit_prompts, device)
    del target_model
    torch.cuda.empty_cache()

    print(f"  Pruned: {pruned_res.shape}, Target: {target_res.shape}")

    # Fit T
    print(f"Fitting linear probe (ridge={ridge})...")
    T_np = fit_linear_probe(pruned_res, target_res, ridge=ridge)
    # Diagnostic: reconstruction error
    reconstructed = pruned_res @ T_np
    err = np.linalg.norm(reconstructed - target_res) / np.linalg.norm(target_res)
    print(f"  Relative reconstruction error: {err:.4f}")
    print(f"  T shape: {T_np.shape}, frobenius norm: {np.linalg.norm(T_np):.4f}")

    # Load pruned model and install hook
    print("\nInstalling linear probe hook on pruned model...")
    fixed_model, fixed_tok = load_model(PRUNED_MODEL, device)
    T = torch.tensor(T_np, dtype=torch.float32, device=device)
    hook = LinearProbeHook(T)
    handle = fixed_model.model.norm.register_forward_pre_hook(hook, with_kwargs=True)

    try:
        print("Evaluating...")
        scores = evaluate_model(fixed_model, fixed_tok, sources, references,
                                max_eval_samples)
    finally:
        handle.remove()

    print(f"  COMET={scores['comet']:.4f}, chrF={scores['chrf']:.2f}, "
          f"BLEU={scores['bleu']:.2f}")

    # Save T for reproducibility
    probe_dir = RESULTS_DIR / "linear_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    np.save(probe_dir / "T.npy", T_np)

    del fixed_model
    torch.cuda.empty_cache()

    return {
        "approach": "linear_probe",
        "reconstruction_error": float(err),
        "T_frobenius_norm": float(np.linalg.norm(T_np)),
        "ridge": ridge,
        "n_fit_samples": n_samples,
        **scores,
    }


# ──────────────────────────────────────────────────────────────────────────
# Approach 3: Fine-tune only LM head + final norm
# ──────────────────────────────────────────────────────────────────────────


def approach_lm_head_ft(device: str = "cuda", n_train: int = 5000,
                       max_eval_samples: int = 200, lr: float = 5e-4,
                       epochs: int = 1, batch_size: int = 4):
    """Train only lm_head and final norm. Freeze everything else."""
    print("=== Approach 3: LM Head + Final Norm FT ===")
    set_seed(42)

    # Load data
    with open(TRAIN_EN) as f:
        train_src = [l.strip() for l in f if l.strip()][:n_train]
    with open(TRAIN_ES) as f:
        train_tgt = [l.strip() for l in f if l.strip()][:n_train]

    texts = [format_prompt(s) + " " + t for s, t in zip(train_src, train_tgt)]
    dataset = Dataset.from_dict({"text": texts})
    print(f"Training on {len(dataset)} examples")

    with open(TEST_EN) as f:
        sources = [l.strip() for l in f if l.strip()]
    with open(TEST_ES) as f:
        references = [l.strip() for l in f if l.strip()]

    # Load pruned model
    print("Loading pruned model...")
    tokenizer = AutoTokenizer.from_pretrained(PRUNED_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        PRUNED_MODEL, torch_dtype=torch.bfloat16, device_map=device
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Freeze everything except lm_head and final norm
    for p in model.parameters():
        p.requires_grad = False
    for p in model.lm_head.parameters():
        p.requires_grad = True
    for p in model.model.norm.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Training
    output_dir = RESULTS_DIR / "lm_head_ft"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        max_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    print("Training...")
    trainer.train()

    # Evaluate
    model.eval()
    print("Evaluating...")
    scores = evaluate_model(model, tokenizer, sources, references, max_eval_samples)
    print(f"  COMET={scores['comet']:.4f}, chrF={scores['chrf']:.2f}, "
          f"BLEU={scores['bleu']:.2f}")

    # Cleanup trainer dir
    import shutil
    shutil.rmtree(output_dir / "trainer", ignore_errors=True)

    del model
    torch.cuda.empty_cache()

    return {
        "approach": "lm_head_ft",
        "trainable_params": trainable,
        "trainable_pct": 100 * trainable / total,
        "n_train_examples": n_train,
        "epochs": epochs,
        **scores,
    }


# ──────────────────────────────────────────────────────────────────────────
# Approach 4: Fine-tune only last-K MLPs
# ──────────────────────────────────────────────────────────────────────────


def approach_mlp_last_ft(device: str = "cuda", n_train: int = 5000,
                         max_eval_samples: int = 200, lr: float = 1e-4,
                         epochs: int = 1, batch_size: int = 4,
                         last_k_layers: int = 3):
    """Train only MLPs in the last K layers. Freeze everything else."""
    print(f"=== Approach 4: Last-{last_k_layers} MLP FT ===")
    set_seed(42)

    with open(TRAIN_EN) as f:
        train_src = [l.strip() for l in f if l.strip()][:n_train]
    with open(TRAIN_ES) as f:
        train_tgt = [l.strip() for l in f if l.strip()][:n_train]

    texts = [format_prompt(s) + " " + t for s, t in zip(train_src, train_tgt)]
    dataset = Dataset.from_dict({"text": texts})
    print(f"Training on {len(dataset)} examples")

    with open(TEST_EN) as f:
        sources = [l.strip() for l in f if l.strip()]
    with open(TEST_ES) as f:
        references = [l.strip() for l in f if l.strip()]

    print("Loading pruned model...")
    tokenizer = AutoTokenizer.from_pretrained(PRUNED_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        PRUNED_MODEL, torch_dtype=torch.bfloat16, device_map=device
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Freeze everything, then unfreeze last-K MLPs
    for p in model.parameters():
        p.requires_grad = False
    n_layers = len(model.model.layers)
    for i in range(n_layers - last_k_layers, n_layers):
        for p in model.model.layers[i].mlp.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    output_dir = RESULTS_DIR / "mlp_last_ft"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        max_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    print("Training...")
    trainer.train()

    model.eval()
    print("Evaluating...")
    scores = evaluate_model(model, tokenizer, sources, references, max_eval_samples)
    print(f"  COMET={scores['comet']:.4f}, chrF={scores['chrf']:.2f}, "
          f"BLEU={scores['bleu']:.2f}")

    import shutil
    shutil.rmtree(output_dir / "trainer", ignore_errors=True)

    del model
    torch.cuda.empty_cache()

    return {
        "approach": "mlp_last_ft",
        "last_k_layers": last_k_layers,
        "trainable_params": trainable,
        "trainable_pct": 100 * trainable / total,
        "n_train_examples": n_train,
        "epochs": epochs,
        **scores,
    }


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approach",
        choices=["norm_rescale", "linear_probe", "lm_head_ft", "mlp_last_ft", "all"],
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Samples for residual collection (approaches 1, 2)")
    parser.add_argument("--n-train", type=int, default=5000,
                        help="Training examples (approaches 3, 4)")
    parser.add_argument("--max-eval-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--last-k-layers", type=int, default=3)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    if args.approach in ("norm_rescale", "all"):
        results.append(approach_norm_rescale(
            args.device, args.n_samples, args.max_eval_samples))
    if args.approach in ("linear_probe", "all"):
        results.append(approach_linear_probe(
            args.device, args.n_samples, args.max_eval_samples))
    if args.approach in ("lm_head_ft", "all"):
        results.append(approach_lm_head_ft(
            args.device, args.n_train, args.max_eval_samples,
            epochs=args.epochs))
    if args.approach in ("mlp_last_ft", "all"):
        results.append(approach_mlp_last_ft(
            args.device, args.n_train, args.max_eval_samples,
            epochs=args.epochs, last_k_layers=args.last_k_layers))

    # Save results
    out_path = RESULTS_DIR / f"surgical_{args.approach}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"{'Approach':<20} {'COMET':<10} {'chrF':<10} {'BLEU':<10}")
    print("-" * 55)
    for r in results:
        print(f"{r['approach']:<20} {r['comet']:<10.4f} {r['chrf']:<10.2f} {r['bleu']:<10.2f}")


if __name__ == "__main__":
    main()
