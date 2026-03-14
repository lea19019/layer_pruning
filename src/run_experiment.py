"""Master experiment runner. Executes a single experiment from a YAML config.

Usage:
    python -m src.run_experiment experiments/configs/I1.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.config import BASE_MODEL, FILTERED_DIR, KD_DIR, RESULTS_DIR
from src.utils import ensure_dir, load_env, set_seed


def run_experiment(config_path: Path):
    """Run a single experiment defined by a YAML config file."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_id = cfg["experiment_id"]
    print(f"\n{'='*60}")
    print(f"Running experiment: {exp_id} - {cfg.get('description', '')}")
    print(f"{'='*60}\n")

    load_env()
    set_seed(cfg.get("seed", 42))

    model_path = cfg.get("base_model", BASE_MODEL)
    exp_dir = ensure_dir(RESULTS_DIR / exp_id)

    # ── Step 1: Pruning ──────────────────────────────────────────────────
    pruning_cfg = cfg.get("pruning", {})
    pruning_method = pruning_cfg.get("method", "none")

    if pruning_method == "none":
        print("No pruning.")
    elif pruning_method == "heuristic":
        from src.pruning.heuristic import iterative_prune
        from src.pruning.remove_layers import load_and_prune, save_pruned_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        target_layers = pruning_cfg["target_layers"]
        val_size = pruning_cfg.get("val_size", 50)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float16, device_map="auto"
        )

        # Load validation data
        with open(FILTERED_DIR / "test.cs") as f:
            val_src = f.read().splitlines()[:val_size]
        with open(FILTERED_DIR / "test.de") as f:
            val_ref = f.read().splitlines()[:val_size]

        removed = iterative_prune(
            model, tokenizer, val_src, val_ref,
            target_layers=target_layers,
            log_path=exp_dir / "pruning_log.json",
        )

        pruned_dir = exp_dir / "pruned_model"
        save_pruned_model(model, tokenizer, str(pruned_dir))
        model_path = str(pruned_dir)

    elif pruning_method == "ifr":
        from src.pruning.guided import get_pruning_plan
        from src.pruning.remove_layers import load_and_prune, save_pruned_model

        scores_path = Path(pruning_cfg.get("scores_path", RESULTS_DIR / "ifr_scores.json"))
        n_remove = pruning_cfg.get("n_remove")
        threshold_factor = pruning_cfg.get("threshold_factor")

        layers_to_remove = get_pruning_plan(scores_path, n_remove, threshold_factor)

        model, tokenizer = load_and_prune(model_path, layers_to_remove)
        pruned_dir = exp_dir / "pruned_model"
        save_pruned_model(model, tokenizer, str(pruned_dir))
        model_path = str(pruned_dir)

        # Save pruning info
        with open(exp_dir / "pruning_info.json", "w") as f:
            json.dump({"layers_removed": layers_to_remove, "method": "ifr"}, f, indent=2)

    elif pruning_method == "lrp":
        # Placeholder for LRP pruning (secondary, time permitting)
        print("LRP pruning not yet implemented.")
        return

    # ── Step 2: Fine-tuning ──────────────────────────────────────────────
    ft_cfg = cfg.get("finetuning", {})
    do_ft = ft_cfg.get("enabled", False)
    do_kd = cfg.get("distillation", {}).get("enabled", False)

    if do_ft:
        use_qlora = ft_cfg.get("qlora", False)

        if do_kd:
            from src.distillation.train_kd import finetune_with_kd
            ft_output = exp_dir / "finetuned"
            model_path = str(finetune_with_kd(model_path, ft_output, use_qlora=use_qlora))
        else:
            from src.finetuning.train import finetune
            ft_output = exp_dir / "finetuned"
            model_path = str(finetune(
                model_name_or_path=model_path,
                train_src=FILTERED_DIR / "train.cs",
                train_tgt=FILTERED_DIR / "train.de",
                output_dir=ft_output,
                use_qlora=use_qlora,
                epochs=ft_cfg.get("epochs", 3),
            ))

    # ── Step 3: Quantization ─────────────────────────────────────────────
    quant_cfg = cfg.get("quantization", {})
    do_quant = quant_cfg.get("enabled", False)

    if do_quant:
        from src.quantization.quantize import quantize_model
        quant_dir = exp_dir / "quantized"
        model_path = str(quantize_model(model_path, quant_dir))

    # ── Step 4: Evaluation ───────────────────────────────────────────────
    from src.config import SRC_LANG_NAME, TGT_LANG_NAME, TRANSLATION_PROMPT
    from src.evaluation.metrics import evaluate_all
    from src.evaluation.translate import translate_batch

    with open(FILTERED_DIR / "test.cs") as f:
        sources = f.read().splitlines()
    with open(FILTERED_DIR / "test.de") as f:
        references = f.read().splitlines()

    prompts = [
        TRANSLATION_PROMPT.format(
            src_lang=SRC_LANG_NAME, tgt_lang=TGT_LANG_NAME, source=src
        )
        for src in sources
    ]

    # Load final model for evaluation
    from transformers import AutoModelForCausalLM, AutoTokenizer

    eval_kwargs = {"dtype": torch.float16, "device_map": "auto"}
    if do_quant and not do_ft:
        # If only quantization (no separate quant step after FT), load with BnB
        from transformers import BitsAndBytesConfig
        eval_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, **eval_kwargs)

    hypotheses = translate_batch(model, tokenizer, prompts)
    metrics = evaluate_all(hypotheses, references, sources)

    # Save final results
    result = {
        "experiment_id": exp_id,
        "description": cfg.get("description", ""),
        "model_path": model_path,
        "config": cfg,
        "metrics": metrics,
        "n_test": len(sources),
        "num_layers": model.config.num_hidden_layers,
    }
    with open(exp_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Experiment {exp_id} complete!")
    print(f"  COMET:  {metrics['comet']:.4f}")
    print(f"  chrF++: {metrics['chrf']:.2f}")
    print(f"  BLEU:   {metrics['bleu']:.2f}")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Results: {exp_dir / 'results.json'}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Run experiment from config")
    parser.add_argument("config", type=Path, help="Path to experiment YAML config")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
