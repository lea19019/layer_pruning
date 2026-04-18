"""Command-line interface for attention_lp.

Exposes the full prune -> finetune -> quantize -> evaluate pipeline as flags,
with optional YAML config for saved experiment recipes. Flags always override
YAML values when both are provided.

Subcommands:
    run          Run the end-to-end experiment pipeline.
    score-ifr    Compute IFR layer importance scores.
    evaluate     Evaluate an existing model on a test set.
    aggregate    Collect all results.json files into a CSV.

Examples:
    attention-lp run --exp-id my_run --pruning ifr --n-remove 8 --finetune \\
        --qlora --epochs 3 --quantize --bits 4

    attention-lp run --config experiments/configs/I5_t03.yaml

    attention-lp run --config experiments/configs/I5_t03.yaml --lang-pair en-es
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


# ── Flag -> config-dict conversion ──────────────────────────────────────────


def _assign_if_set(cfg: dict, path: tuple[str, ...], value: Any) -> None:
    """Write `value` into nested `cfg` at `path` when `value is not None`."""
    if value is None:
        return
    node = cfg
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def build_config_from_args(args: argparse.Namespace) -> dict:
    """Merge a YAML config (if given) with CLI flag overrides into one dict.

    The returned dict matches the schema that `run_pipeline` expects, with the
    same sections (pruning, finetuning, distillation, quantization) that the
    existing YAML configs use. Flag values override YAML values.
    """
    cfg: dict = {}
    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    _assign_if_set(cfg, ("experiment_id",), args.exp_id)
    _assign_if_set(cfg, ("description",), args.description)
    _assign_if_set(cfg, ("base_model",), args.base_model)
    _assign_if_set(cfg, ("seed",), args.seed)
    _assign_if_set(cfg, ("lang_pair",), args.lang_pair)
    _assign_if_set(cfg, ("data_dir",), args.data_dir)
    _assign_if_set(cfg, ("output_dir",), args.output_dir)
    _assign_if_set(cfg, ("kd_dir",), args.kd_dir)

    # Pruning
    if args.pruning is not None:
        cfg.setdefault("pruning", {})["method"] = args.pruning
    _assign_if_set(cfg, ("pruning", "target_layers"), args.target_layers)
    _assign_if_set(cfg, ("pruning", "n_remove"), args.n_remove)
    _assign_if_set(cfg, ("pruning", "threshold_factor"), args.threshold_factor)
    _assign_if_set(cfg, ("pruning", "scores_path"), args.scores_path)
    _assign_if_set(cfg, ("pruning", "val_size"), args.val_size)
    if args.layers_to_remove is not None:
        cfg.setdefault("pruning", {})["layers_to_remove"] = [
            int(x) for x in args.layers_to_remove.split(",") if x.strip()
        ]

    # Fine-tuning
    if args.finetune:
        cfg.setdefault("finetuning", {})["enabled"] = True
    if args.no_finetune:
        cfg.setdefault("finetuning", {})["enabled"] = False
    if args.qlora:
        cfg.setdefault("finetuning", {})["qlora"] = True
    if args.full_ft:
        cfg.setdefault("finetuning", {})["full_ft"] = True
    _assign_if_set(cfg, ("finetuning", "epochs"), args.epochs)

    # Distillation
    if args.kd:
        cfg.setdefault("distillation", {})["enabled"] = True
    if args.no_kd:
        cfg.setdefault("distillation", {})["enabled"] = False

    # Quantization
    if args.quantize:
        cfg.setdefault("quantization", {})["enabled"] = True
    if args.no_quantize:
        cfg.setdefault("quantization", {})["enabled"] = False
    _assign_if_set(cfg, ("quantization", "bits"), args.bits)
    _assign_if_set(cfg, ("quantization", "quant_type"), args.quant_type)

    # Defaults for sections the YAML normally carries but that may be missing
    # if the user only used flags.
    cfg.setdefault("pruning", {"method": "none"})
    cfg.setdefault("finetuning", {"enabled": False})
    cfg.setdefault("distillation", {"enabled": False})
    cfg.setdefault("quantization", {"enabled": False})

    if "experiment_id" not in cfg:
        raise SystemExit("error: --exp-id is required (or set experiment_id in the YAML config)")

    return cfg


# ── Argparse setup ──────────────────────────────────────────────────────────


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config to load; flags override its values")
    parser.add_argument("--exp-id", "--experiment-id", dest="exp_id", default=None,
                        help="Experiment identifier (used for output dir name)")
    parser.add_argument("--description", default=None)
    parser.add_argument("--base-model", default=None, help="HF model name or local path")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lang-pair", choices=["cs-de", "en-es"], default=None,
                        help="Shortcut that selects source/target language and data dir")
    parser.add_argument("--data-dir", default=None,
                        help="Override data dir (expects test.<src>/test.<tgt>, train.<src>/train.<tgt>)")
    parser.add_argument("--output-dir", default=None,
                        help="Override experiment output dir (default: experiments/results/<exp-id>)")
    parser.add_argument("--kd-dir", default=None,
                        help="Override KD data dir (only used when --kd is set)")

    p = parser.add_argument_group("pruning")
    p.add_argument("--pruning", choices=["none", "heuristic", "ifr", "lrp"], default=None,
                   help="Pruning method")
    p.add_argument("--target-layers", type=int, default=None,
                   help="[heuristic] number of layers to KEEP after pruning")
    p.add_argument("--n-remove", type=int, default=None,
                   help="[ifr] number of layers to remove")
    p.add_argument("--threshold-factor", type=float, default=None,
                   help="[ifr] remove layers with importance < threshold * max")
    p.add_argument("--scores-path", default=None,
                   help="[ifr] path to precomputed IFR scores JSON")
    p.add_argument("--val-size", type=int, default=None,
                   help="[heuristic] number of validation sentences")
    p.add_argument("--layers-to-remove", default=None,
                   help="Comma-separated layer indices to remove (reproducibility)")

    f = parser.add_argument_group("finetuning")
    f.add_argument("--finetune", action="store_true", help="Enable fine-tuning")
    f.add_argument("--no-finetune", action="store_true", help="Disable fine-tuning")
    f.add_argument("--qlora", action="store_true", help="Use QLoRA")
    f.add_argument("--full-ft", action="store_true", help="Full fine-tuning (not LoRA)")
    f.add_argument("--epochs", type=int, default=None)
    f.add_argument("--kd", action="store_true", help="Enable knowledge distillation")
    f.add_argument("--no-kd", action="store_true")

    q = parser.add_argument_group("quantization")
    q.add_argument("--quantize", action="store_true", help="Enable quantization")
    q.add_argument("--no-quantize", action="store_true")
    q.add_argument("--bits", type=int, choices=[4, 8], default=None)
    q.add_argument("--quant-type", choices=["nf4", "fp4"], default=None)


def _cmd_run(args: argparse.Namespace) -> None:
    from src.run_experiment import run_pipeline

    cfg = build_config_from_args(args)
    run_pipeline(cfg)


def _cmd_score_ifr(args: argparse.Namespace) -> None:
    import sys
    from src.attribution import score_layers

    forwarded = ["--model", args.model, "--src", str(args.src), "--tgt", str(args.tgt),
                 "--n-samples", str(args.n_samples), "--max-length", str(args.max_length),
                 "--output", str(args.output), "--seed", str(args.seed),
                 "--src-lang-name", args.src_lang_name, "--tgt-lang-name", args.tgt_lang_name]
    sys.argv = ["score-ifr"] + forwarded
    score_layers.main()


def _cmd_evaluate(args: argparse.Namespace) -> None:
    import sys
    from src.evaluation import run_eval

    forwarded = ["--model", args.model, "--test-src", str(args.test_src),
                 "--test-tgt", str(args.test_tgt), "--batch-size", str(args.batch_size),
                 "--max-new-tokens", str(args.max_new_tokens)]
    if args.output is not None:
        forwarded += ["--output", str(args.output)]
    if args.experiment_id is not None:
        forwarded += ["--experiment-id", args.experiment_id]
    if args.use_vllm:
        forwarded += ["--use-vllm", "--tp-size", str(args.tp_size)]
    sys.argv = ["evaluate"] + forwarded
    run_eval.main()


def _cmd_aggregate(args: argparse.Namespace) -> None:
    from src.evaluation.aggregate_results import collect_results, print_table

    df = collect_results(args.results_dir)
    print_table(df)
    out = args.output if args.output is not None else args.results_dir / "all_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="attention-lp",
        description="Layer pruning + fine-tuning + quantization + evaluation pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the end-to-end pipeline")
    _add_run_args(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_score = sub.add_parser("score-ifr", help="Compute IFR layer importance scores")
    # Defaults are resolved at call time (imports src.config then).
    p_score.add_argument("--model", default=None)
    p_score.add_argument("--src", type=Path, default=None)
    p_score.add_argument("--tgt", type=Path, default=None)
    p_score.add_argument("--n-samples", type=int, default=200)
    p_score.add_argument("--max-length", type=int, default=256)
    p_score.add_argument("--output", type=Path, default=None)
    p_score.add_argument("--seed", type=int, default=42)
    p_score.add_argument("--src-lang-name", default=None)
    p_score.add_argument("--tgt-lang-name", default=None)
    p_score.set_defaults(func=_cmd_score_ifr)

    p_eval = sub.add_parser("evaluate", help="Evaluate a model on a test set")
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--test-src", type=Path, required=True)
    p_eval.add_argument("--test-tgt", type=Path, required=True)
    p_eval.add_argument("--output", type=Path, default=None)
    p_eval.add_argument("--use-vllm", action="store_true")
    p_eval.add_argument("--tp-size", type=int, default=1)
    p_eval.add_argument("--batch-size", type=int, default=8)
    p_eval.add_argument("--max-new-tokens", type=int, default=256)
    p_eval.add_argument("--experiment-id", default=None)
    p_eval.set_defaults(func=_cmd_evaluate)

    p_agg = sub.add_parser("aggregate", help="Aggregate experiment results into a CSV")
    p_agg.add_argument("--results-dir", type=Path, default=None,
                       help="Defaults to experiments/results/")
    p_agg.add_argument("--output", type=Path, default=None)
    p_agg.set_defaults(func=_cmd_aggregate)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve defaults that depend on src.config (imported lazily so --help is fast).
    if args.command == "score-ifr":
        from src.config import BASE_MODEL, FILTERED_DIR, RESULTS_DIR, SRC_LANG_NAME, TGT_LANG_NAME
        if args.model is None:
            args.model = BASE_MODEL
        if args.src is None:
            args.src = FILTERED_DIR / "test.cs"
        if args.tgt is None:
            args.tgt = FILTERED_DIR / "test.de"
        if args.output is None:
            args.output = RESULTS_DIR / "ifr_scores.json"
        if args.src_lang_name is None:
            args.src_lang_name = SRC_LANG_NAME
        if args.tgt_lang_name is None:
            args.tgt_lang_name = TGT_LANG_NAME
    elif args.command == "aggregate":
        if args.results_dir is None:
            from src.config import RESULTS_DIR
            args.results_dir = RESULTS_DIR

    args.func(args)


if __name__ == "__main__":
    main()
