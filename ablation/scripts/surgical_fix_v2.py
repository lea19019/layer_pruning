"""Distributed surgical fixes for the pruned model.

Approaches 5-8 test the hypothesis that distribution (not parameter count)
is what makes LoRA recovery work. Each approach places corrections at EVERY
layer, but with different constraints:

  5. per_layer_norm: Scale each RMSNorm weight by per-layer ratio of
     pruned/target residual RMS. 17 scalars total, zero training.

  6. bias_only: Add a learnable bias vector b_l ∈ ℝ^d at the output of
     every layer. Train with SFT. ~66K trainable params (16 × 4096).

  7. procrustes: At every layer, fit orthogonal rotation R_l via SVD to
     align pruned residual to target. Closed-form, no training.

  8. low_rank_probes: At every layer, fit a rank-r linear correction via
     reduced-rank regression. Closed-form, no training. r=16 default
     (matching original LoRA).

Usage:
    python -m ablation.scripts.surgical_fix_v2 --approach per_layer_norm
    python -m ablation.scripts.surgical_fix_v2 --approach bias_only
    python -m ablation.scripts.surgical_fix_v2 --approach procrustes
    python -m ablation.scripts.surgical_fix_v2 --approach low_rank_probes --rank 16
    python -m ablation.scripts.surgical_fix_v2 --approach all
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
RESULTS_DIR = PROJECT_ROOT / "ablation/results/surgical_fix_v2"

PRUNED_MODEL = str(PROJECT_ROOT / "experiments/results/IP_16_enes/pruned_model")
TARGET_MODEL = str(
    PROJECT_ROOT / "experiments/results/I2_16_enes/finetuned/merged"
)

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


def collect_residuals_at_all_norms(
    model, tokenizer, prompts: list[str], device: str = "cuda",
    batch_size: int = 4,
) -> list[np.ndarray]:
    """Collect pre-norm residuals at each RMSNorm position.

    Cohere architecture has one input_layernorm per layer + one final norm.
    For n_layers layers, there are n_layers + 1 norm positions.

    Returns:
        List of (n_prompts, d_model) arrays, one per norm, last-token-only.
    """
    layers = model.model.layers
    norm_modules = [layer.input_layernorm for layer in layers] + [model.model.norm]
    n_positions = len(norm_modules)
    d_model = model.config.hidden_size

    residuals = [[] for _ in range(n_positions)]
    captured = {}
    hooks = []

    for i, norm in enumerate(norm_modules):
        def make_hook(idx):
            def hook(module, args, kwargs):
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is not None:
                    captured[idx] = hidden.detach()
            return hook
        hooks.append(norm.register_forward_pre_hook(make_hook(i), with_kwargs=True))

    try:
        for bstart in range(0, len(prompts), batch_size):
            batch = prompts[bstart:bstart + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            ).to(device)
            captured.clear()
            with torch.no_grad():
                model(**inputs)
            seq_lens = inputs["attention_mask"].sum(dim=1) - 1
            for i in range(n_positions):
                h = captured[i]  # (batch, seq, hidden)
                last = h[torch.arange(h.size(0)), seq_lens]
                residuals[i].append(last.cpu().float().numpy())
    finally:
        for h in hooks:
            h.remove()

    return [np.concatenate(r, axis=0) for r in residuals]


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
# Approach 5: Per-layer RMSNorm rescaling (distributed)
# ──────────────────────────────────────────────────────────────────────────


def compute_per_layer_scales(
    pruned_residuals: list[np.ndarray],
    target_residuals: list[np.ndarray],
) -> list[float]:
    """Compute RMS scale factor per norm position."""
    scales = []
    for p, t in zip(pruned_residuals, target_residuals):
        p_rms = np.sqrt((p ** 2).mean())
        t_rms = np.sqrt((t ** 2).mean())
        scales.append(float(t_rms / (p_rms + 1e-8)))
    return scales


def approach_per_layer_norm(device: str = "cuda", n_samples: int = 500,
                            max_eval_samples: int = 200):
    """Rescale each RMSNorm weight by per-layer RMS ratio."""
    print("=== Approach 5: Per-Layer Norm Rescale ===")

    with open(TEST_EN) as f:
        sources = [l.strip() for l in f if l.strip()]
    with open(TEST_ES) as f:
        references = [l.strip() for l in f if l.strip()]

    fit_prompts = [format_prompt(s) for s in sources[:n_samples]]

    print(f"Collecting residuals at all norm positions...")
    pruned_model, pruned_tok = load_model(PRUNED_MODEL, device)
    pruned_res = collect_residuals_at_all_norms(pruned_model, pruned_tok, fit_prompts, device)
    del pruned_model
    torch.cuda.empty_cache()

    target_model, target_tok = load_model(TARGET_MODEL, device)
    target_res = collect_residuals_at_all_norms(target_model, target_tok, fit_prompts, device)
    del target_model
    torch.cuda.empty_cache()

    n_positions = len(pruned_res)
    print(f"  {n_positions} norm positions, {pruned_res[0].shape[0]} prompts each")

    scales = compute_per_layer_scales(pruned_res, target_res)
    print(f"  Scale factors (min={min(scales):.4f}, max={max(scales):.4f}, mean={np.mean(scales):.4f})")
    for i, s in enumerate(scales):
        tag = f"layer_{i}.input_layernorm" if i < n_positions - 1 else "model.norm"
        print(f"    {tag}: scale={s:.4f}")

    print("\nApplying scales...")
    fixed_model, fixed_tok = load_model(PRUNED_MODEL, device)
    with torch.no_grad():
        for i, layer in enumerate(fixed_model.model.layers):
            layer.input_layernorm.weight.data *= scales[i]
        fixed_model.model.norm.weight.data *= scales[-1]

    print("Evaluating...")
    scores = evaluate_model(fixed_model, fixed_tok, sources, references, max_eval_samples)
    print(f"  COMET={scores['comet']:.4f}, chrF={scores['chrf']:.2f}, BLEU={scores['bleu']:.2f}")

    del fixed_model
    torch.cuda.empty_cache()

    return {
        "approach": "per_layer_norm",
        "n_positions": n_positions,
        "scales": scales,
        "n_collection_samples": n_samples,
        **scores,
    }


# ──────────────────────────────────────────────────────────────────────────
# Approach 6: Bias-only FT (distributed, minimal)
# ──────────────────────────────────────────────────────────────────────────


def _attach_layer_biases(model, d_model: int, n_layers: int):
    """Attach learnable bias parameters + hooks that add them to layer outputs.

    Returns the ParameterList and list of hook handles.
    """
    # Parameters in float32 for stable gradients; will be cast to bf16 at use time
    biases = nn.ParameterList([
        nn.Parameter(torch.zeros(d_model, dtype=torch.float32))
        for _ in range(n_layers)
    ])
    model.layer_biases = biases  # attach so model.parameters() picks them up

    handles = []
    for i, layer in enumerate(model.model.layers):
        bias_param = biases[i]

        def make_hook(bp):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                # Cast bias to compute dtype/device
                b = bp.to(device=hidden.device, dtype=hidden.dtype)
                hidden = hidden + b
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
            return hook

        handles.append(layer.register_forward_hook(make_hook(bias_param)))

    return biases, handles


def approach_bias_only(device: str = "cuda", n_train: int = 5000,
                      max_eval_samples: int = 200, lr: float = 1e-3,
                      epochs: int = 1, batch_size: int = 4):
    """Train only per-layer bias vectors. Freeze everything else."""
    print("=== Approach 6: Bias-Only FT ===")
    set_seed(42)

    with open(TRAIN_EN) as f:
        train_src = [l.strip() for l in f if l.strip()][:n_train]
    with open(TRAIN_ES) as f:
        train_tgt = [l.strip() for l in f if l.strip()][:n_train]
    texts = [format_prompt(s) + " " + t for s, t in zip(train_src, train_tgt)]
    dataset = Dataset.from_dict({"text": texts})

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

    d_model = model.config.hidden_size
    n_layers = len(model.model.layers)

    # Attach biases with hooks
    biases, handles = _attach_layer_biases(model, d_model, n_layers)
    biases.to(device)

    # Freeze everything except biases
    for p in model.parameters():
        p.requires_grad = False
    for p in biases.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    print(f"  Training on {len(dataset)} examples, lr={lr}, epochs={epochs}")

    output_dir = RESULTS_DIR / "bias_only"
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
    print(f"  COMET={scores['comet']:.4f}, chrF={scores['chrf']:.2f}, BLEU={scores['bleu']:.2f}")

    # Save biases for reproducibility
    bias_np = np.stack([b.detach().cpu().float().numpy() for b in biases])
    np.save(output_dir / "learned_biases.npy", bias_np)

    # Cleanup
    import shutil
    shutil.rmtree(output_dir / "trainer", ignore_errors=True)
    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()

    return {
        "approach": "bias_only",
        "trainable_params": trainable,
        "trainable_pct": 100 * trainable / total,
        "bias_norms": [float(np.linalg.norm(b)) for b in bias_np],
        "n_train_examples": n_train,
        "epochs": epochs,
        **scores,
    }


# ──────────────────────────────────────────────────────────────────────────
# Approach 7: Layer-wise Procrustes (orthogonal alignment)
# ──────────────────────────────────────────────────────────────────────────


def orthogonal_procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Solve min ||X @ R - Y||_F over orthogonal R.

    Returns:
        R: (d, d) orthogonal matrix.
    """
    M = X.T @ Y
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def _collect_layer_outputs(model, tokenizer, prompts, device, batch_size=4):
    """Collect output (post-residual) of each layer. Used for 7, 8."""
    layers = model.model.layers
    captured = {}
    hooks = []

    for i, layer in enumerate(layers):
        def make_hook(idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[idx] = hidden.detach()
            return hook
        hooks.append(layer.register_forward_hook(make_hook(i)))

    outputs = [[] for _ in range(len(layers))]
    try:
        for bstart in range(0, len(prompts), batch_size):
            batch = prompts[bstart:bstart + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            ).to(device)
            captured.clear()
            with torch.no_grad():
                model(**inputs)
            seq_lens = inputs["attention_mask"].sum(dim=1) - 1
            for i in range(len(layers)):
                h = captured[i]
                last = h[torch.arange(h.size(0)), seq_lens]
                outputs[i].append(last.cpu().float().numpy())
    finally:
        for h in hooks:
            h.remove()

    return [np.concatenate(o, axis=0) for o in outputs]


def _install_linear_maps(model, Ts: list[torch.Tensor]):
    """Install per-layer linear map hooks: layer_output ← layer_output @ T_l.

    Returns hook handles.
    """
    handles = []
    for i, layer in enumerate(model.model.layers):
        T = Ts[i]
        def make_hook(t):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                orig_dtype = hidden.dtype
                fixed = (hidden.float() @ t).to(orig_dtype)
                if isinstance(output, tuple):
                    return (fixed,) + output[1:]
                return fixed
            return hook
        handles.append(layer.register_forward_hook(make_hook(T)))
    return handles


def approach_procrustes(device: str = "cuda", n_samples: int = 1000,
                       max_eval_samples: int = 200):
    """Fit orthogonal rotation at every layer; install via hooks."""
    print("=== Approach 7: Layer-wise Procrustes ===")

    with open(TEST_EN) as f:
        sources = [l.strip() for l in f if l.strip()]
    with open(TEST_ES) as f:
        references = [l.strip() for l in f if l.strip()]
    with open(TRAIN_EN) as f:
        train_sources = [l.strip() for l in f if l.strip()]
    fit_prompts = [format_prompt(s) for s in train_sources[:n_samples]]

    print(f"Collecting layer outputs on {n_samples} prompts from each model...")
    pruned_model, pruned_tok = load_model(PRUNED_MODEL, device)
    pruned_outs = _collect_layer_outputs(pruned_model, pruned_tok, fit_prompts, device)
    del pruned_model
    torch.cuda.empty_cache()

    target_model, target_tok = load_model(TARGET_MODEL, device)
    target_outs = _collect_layer_outputs(target_model, target_tok, fit_prompts, device)
    del target_model
    torch.cuda.empty_cache()

    n_layers = len(pruned_outs)
    print(f"  {n_layers} layers, {pruned_outs[0].shape[0]} samples each")

    print("Fitting orthogonal rotations per layer...")
    Rs = []
    errors = []
    for i in range(n_layers):
        R = orthogonal_procrustes(pruned_outs[i], target_outs[i])
        err = np.linalg.norm(pruned_outs[i] @ R - target_outs[i]) / np.linalg.norm(target_outs[i])
        Rs.append(R)
        errors.append(float(err))
        print(f"  Layer {i:2d}: rel_err = {err:.4f}")

    print(f"\nMean reconstruction error: {np.mean(errors):.4f}")

    print("Installing rotations on pruned model...")
    fixed_model, fixed_tok = load_model(PRUNED_MODEL, device)
    R_tensors = [torch.tensor(R, dtype=torch.float32, device=device) for R in Rs]
    handles = _install_linear_maps(fixed_model, R_tensors)

    try:
        print("Evaluating...")
        scores = evaluate_model(fixed_model, fixed_tok, sources, references, max_eval_samples)
    finally:
        for h in handles:
            h.remove()

    print(f"  COMET={scores['comet']:.4f}, chrF={scores['chrf']:.2f}, BLEU={scores['bleu']:.2f}")

    # Save rotations
    out_dir = RESULTS_DIR / "procrustes"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "rotations.npz",
                        **{f"layer_{i}": R for i, R in enumerate(Rs)})

    del fixed_model
    torch.cuda.empty_cache()

    return {
        "approach": "procrustes",
        "n_layers": n_layers,
        "reconstruction_errors": errors,
        "mean_error": float(np.mean(errors)),
        "n_fit_samples": n_samples,
        **scores,
    }


# ──────────────────────────────────────────────────────────────────────────
# Approach 8: Distributed low-rank probes (reduced-rank regression)
# ──────────────────────────────────────────────────────────────────────────


def reduced_rank_regression(
    X: np.ndarray, Y: np.ndarray, rank: int, ridge: float = 1e-3
) -> np.ndarray:
    """Fit X @ T ≈ Y with rank(T) ≤ rank, via SVD-truncated ridge.

    Returns:
        T: (d, d) rank-r linear map.
    """
    d = X.shape[1]
    # Full-rank ridge solution
    T_full = np.linalg.solve(X.T @ X + ridge * np.eye(d), X.T @ Y)
    # Truncate to rank r via SVD
    U, S, Vt = np.linalg.svd(T_full, full_matrices=False)
    S_r = np.zeros_like(S)
    S_r[:rank] = S[:rank]
    T_r = U @ np.diag(S_r) @ Vt
    return T_r


def approach_low_rank_probes(device: str = "cuda", n_samples: int = 2000,
                             max_eval_samples: int = 200, rank: int = 16,
                             ridge: float = 1e-3):
    """Fit rank-r linear correction at every layer via closed-form RRR."""
    print(f"=== Approach 8: Distributed Low-Rank Probes (r={rank}) ===")

    with open(TEST_EN) as f:
        sources = [l.strip() for l in f if l.strip()]
    with open(TEST_ES) as f:
        references = [l.strip() for l in f if l.strip()]
    with open(TRAIN_EN) as f:
        train_sources = [l.strip() for l in f if l.strip()]
    fit_prompts = [format_prompt(s) for s in train_sources[:n_samples]]

    print(f"Collecting layer outputs on {n_samples} prompts...")
    pruned_model, pruned_tok = load_model(PRUNED_MODEL, device)
    pruned_outs = _collect_layer_outputs(pruned_model, pruned_tok, fit_prompts, device)
    del pruned_model
    torch.cuda.empty_cache()

    target_model, target_tok = load_model(TARGET_MODEL, device)
    target_outs = _collect_layer_outputs(target_model, target_tok, fit_prompts, device)
    del target_model
    torch.cuda.empty_cache()

    n_layers = len(pruned_outs)

    print(f"Fitting rank-{rank} maps per layer...")
    # We want: X_l @ (I + delta_l) = Y_l, so delta_l = T_l - I
    # Using T_l directly (not delta): Y ≈ X @ T
    Ts = []
    errors = []
    for i in range(n_layers):
        T = reduced_rank_regression(pruned_outs[i], target_outs[i], rank, ridge)
        err = np.linalg.norm(pruned_outs[i] @ T - target_outs[i]) / np.linalg.norm(target_outs[i])
        Ts.append(T)
        errors.append(float(err))
        print(f"  Layer {i:2d}: rel_err = {err:.4f}, T frob = {np.linalg.norm(T):.2f}")

    print(f"\nMean reconstruction error: {np.mean(errors):.4f}")

    print("Installing rank-{rank} corrections on pruned model...")
    fixed_model, fixed_tok = load_model(PRUNED_MODEL, device)
    T_tensors = [torch.tensor(T, dtype=torch.float32, device=device) for T in Ts]
    handles = _install_linear_maps(fixed_model, T_tensors)

    try:
        print("Evaluating...")
        scores = evaluate_model(fixed_model, fixed_tok, sources, references, max_eval_samples)
    finally:
        for h in handles:
            h.remove()

    print(f"  COMET={scores['comet']:.4f}, chrF={scores['chrf']:.2f}, BLEU={scores['bleu']:.2f}")

    out_dir = RESULTS_DIR / f"low_rank_probes_r{rank}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "probes.npz",
                        **{f"layer_{i}": T for i, T in enumerate(Ts)})

    del fixed_model
    torch.cuda.empty_cache()

    return {
        "approach": "low_rank_probes",
        "rank": rank,
        "ridge": ridge,
        "n_layers": n_layers,
        "reconstruction_errors": errors,
        "mean_error": float(np.mean(errors)),
        "n_fit_samples": n_samples,
        **scores,
    }


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approach",
        choices=["per_layer_norm", "bias_only", "procrustes", "low_rank_probes", "all"],
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Samples for residual collection (approaches 5, 7, 8)")
    parser.add_argument("--n-train", type=int, default=5000,
                        help="Training examples (approach 6)")
    parser.add_argument("--max-eval-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, default=16,
                        help="Rank for low_rank_probes")
    parser.add_argument("--ridge", type=float, default=1e-3,
                        help="Ridge regularization for low_rank_probes")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    if args.approach in ("per_layer_norm", "all"):
        results.append(approach_per_layer_norm(
            args.device, args.n_samples, args.max_eval_samples))
    if args.approach in ("procrustes", "all"):
        results.append(approach_procrustes(
            args.device, args.n_samples, args.max_eval_samples))
    if args.approach in ("low_rank_probes", "all"):
        results.append(approach_low_rank_probes(
            args.device, args.n_samples, args.max_eval_samples,
            rank=args.rank, ridge=args.ridge))
    if args.approach in ("bias_only", "all"):
        results.append(approach_bias_only(
            args.device, args.n_train, args.max_eval_samples,
            epochs=args.epochs))

    out_path = RESULTS_DIR / f"surgical_v2_{args.approach}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n=== Summary ===")
    print(f"{'Approach':<22} {'COMET':<10} {'chrF':<10} {'BLEU':<10}")
    print("-" * 55)
    for r in results:
        print(f"{r['approach']:<22} {r['comet']:<10.4f} {r['chrf']:<10.2f} {r['bleu']:<10.2f}")


if __name__ == "__main__":
    main()
