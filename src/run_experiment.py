"""Master experiment runner. Executes a single experiment from a YAML config.

Usage:
    python -m src.run_experiment experiments/configs/I1.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.config import BASE_MODEL, FILTERED_DIR, KD_DIR, RESULTS_DIR, TRANSLATION_PROMPT
from src.utils import ensure_dir, load_env, set_seed


# Language pair defaults and mappings
LANG_DEFAULTS = {
    "cs-de": {"src_code": "cs", "tgt_code": "de", "src_name": "Czech", "tgt_name": "German"},
    "en-es": {"src_code": "en", "tgt_code": "es", "src_name": "English", "tgt_name": "Spanish"},
}


def _resolve_lang_pair(cfg: dict) -> dict:
    """Resolve language pair settings from config.

    Supports either explicit src_lang/tgt_lang fields or a lang_pair shortcut.
    Defaults to cs-de for backward compatibility.
    """
    lang_pair = cfg.get("lang_pair", "cs-de")
    defaults = LANG_DEFAULTS.get(lang_pair, LANG_DEFAULTS["cs-de"])

    return {
        "src_code": cfg.get("src_lang_code", defaults["src_code"]),
        "tgt_code": cfg.get("tgt_lang_code", defaults["tgt_code"]),
        "src_name": cfg.get("src_lang_name", defaults["src_name"]),
        "tgt_name": cfg.get("tgt_lang_name", defaults["tgt_name"]),
        "lang_pair": lang_pair,
    }


def _data_dir(lang: dict) -> Path:
    """Get the filtered data directory for a language pair."""
    if lang["lang_pair"] == "cs-de":
        return FILTERED_DIR
    return FILTERED_DIR.parent / f"filtered_{lang['lang_pair'].replace('-', '_')}"


def run_experiment(config_path: Path):
    """Run a single experiment defined by a YAML config file.

    Pipeline: prune -> finetune (optionally with KD) -> quantize -> evaluate.
    Each stage is optional; `model_path` is threaded through so each stage
    picks up the output of the previous one.
    """
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

    # Resolve language pair
    lang = _resolve_lang_pair(cfg)
    data_dir = _data_dir(lang)
    print(f"Language pair: {lang['src_name']} → {lang['tgt_name']} ({lang['lang_pair']})")
    print(f"Data dir: {data_dir}")

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
        val_size = pruning_cfg.get("val_size", 200)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )

        # Load validation data
        with open(data_dir / f"test.{lang['src_code']}") as f:
            val_src = f.read().splitlines()[:val_size]
        with open(data_dir / f"test.{lang['tgt_code']}") as f:
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
        print("LRP pruning not yet implemented.")
        return

    # ── Step 2: Fine-tuning ──────────────────────────────────────────────
    ft_cfg = cfg.get("finetuning", {})
    do_ft = ft_cfg.get("enabled", False)
    do_kd = cfg.get("distillation", {}).get("enabled", False)

    if do_ft:
        use_qlora = ft_cfg.get("qlora", False)
        use_full_ft = ft_cfg.get("full_ft", False)

        if do_kd:
            from src.distillation.train_kd import finetune_with_kd
            ft_output = exp_dir / "finetuned"
            # Resolve KD data directory for this language pair
            kd_dir = KD_DIR if lang["lang_pair"] == "cs-de" else KD_DIR.parent / f"kd_{lang['lang_pair'].replace('-', '_')}"
            model_path = str(finetune_with_kd(
                model_path, ft_output, use_qlora=use_qlora,
                data_dir=data_dir, kd_dir=kd_dir,
                src_ext=lang["src_code"], tgt_ext=lang["tgt_code"],
            ))
        else:
            from src.finetuning.train import finetune
            ft_output = exp_dir / "finetuned"
            model_path = str(finetune(
                model_name_or_path=model_path,
                train_src=data_dir / f"train.{lang['src_code']}",
                train_tgt=data_dir / f"train.{lang['tgt_code']}",
                output_dir=ft_output,
                use_qlora=use_qlora,
                full_ft=use_full_ft,
                epochs=ft_cfg.get("epochs", 3),
            ))

    # ── Step 3: Quantization ─────────────────────────────────────────────
    quant_cfg = cfg.get("quantization", {})
    do_quant = quant_cfg.get("enabled", False)

    if do_quant:
        from src.quantization.quantize import quantize_model
        quant_dir = exp_dir / "quantized"
        quant_bits = quant_cfg.get("bits", 4)
        quant_type = quant_cfg.get("quant_type", "nf4")
        model_path = str(quantize_model(model_path, quant_dir, bits=quant_bits, quant_type=quant_type))

    # ── Step 4: Evaluation ───────────────────────────────────────────────
    from src.evaluation.metrics import evaluate_all
    from src.evaluation.translate import translate_batch

    with open(data_dir / f"test.{lang['src_code']}") as f:
        sources = f.read().splitlines()
    with open(data_dir / f"test.{lang['tgt_code']}") as f:
        references = f.read().splitlines()

    prompts = [
        TRANSLATION_PROMPT.format(
            src_lang=lang["src_name"], tgt_lang=lang["tgt_name"], source=src
        )
        for src in sources
    ]

    # Load final model for evaluation
    from transformers import AutoModelForCausalLM, AutoTokenizer

    eval_kwargs = {"dtype": torch.float16, "device_map": "auto"}
    if do_quant:
        from transformers import BitsAndBytesConfig
        quant_bits = quant_cfg.get("bits", 4)
        if quant_bits == 8:
            eval_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            eval_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
            )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, **eval_kwargs)

    hypotheses = translate_batch(model, tokenizer, prompts)
    metrics = evaluate_all(
        hypotheses, references, sources,
        model=model, tokenizer=tokenizer, prompts=prompts,
    )

    # Save final results
    result = {
        "experiment_id": exp_id,
        "description": cfg.get("description", ""),
        "model_path": model_path,
        "config": cfg,
        "lang_pair": lang["lang_pair"],
        "metrics": metrics,
        "n_test": len(sources),
        "num_layers": model.config.num_hidden_layers,
    }

    # Save sample translations for inspection
    result["sample_translations"] = [
        {"source": s, "reference": r, "hypothesis": h}
        for s, r, h in zip(sources[:10], references[:10], hypotheses[:10])
    ]

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
