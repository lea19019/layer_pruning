"""Generate all experiment YAML configs from the experimental matrix."""

from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

PRUNE_TARGETS = {8: 24, 12: 20, 16: 16}  # n_remove -> target_layers


def write_config(exp_id: str, cfg: dict):
    path = CONFIGS_DIR / f"{exp_id}.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  {path.name}")


# ── Group 2: Moslem replication (heuristic pruning) ──────────────────────────
print("Group 2: Moslem replication")
for n_remove, target in PRUNE_TARGETS.items():
    # M1: Heuristic + FT
    write_config(f"M1_{n_remove}", {
        "experiment_id": f"M1_{n_remove}",
        "description": f"Heuristic pruning ({n_remove} removed -> {target} layers) + fine-tuning",
        "base_model": "CohereForAI/aya-expanse-8b",
        "seed": 42,
        "pruning": {"method": "heuristic", "target_layers": target, "val_size": 200},
        "finetuning": {"enabled": True, "epochs": 3, "qlora": False},
        "distillation": {"enabled": False},
        "quantization": {"enabled": False},
    })
    # M2: Heuristic + FT + KD
    write_config(f"M2_{n_remove}", {
        "experiment_id": f"M2_{n_remove}",
        "description": f"Heuristic pruning ({n_remove} removed) + FT + KD",
        "base_model": "CohereForAI/aya-expanse-8b",
        "seed": 42,
        "pruning": {"method": "heuristic", "target_layers": target, "val_size": 200},
        "finetuning": {"enabled": True, "epochs": 3, "qlora": False},
        "distillation": {"enabled": True},
        "quantization": {"enabled": False},
    })
    # M3: Heuristic + FT + Quant
    write_config(f"M3_{n_remove}", {
        "experiment_id": f"M3_{n_remove}",
        "description": f"Heuristic pruning ({n_remove} removed) + FT + INT4",
        "base_model": "CohereForAI/aya-expanse-8b",
        "seed": 42,
        "pruning": {"method": "heuristic", "target_layers": target, "val_size": 200},
        "finetuning": {"enabled": True, "epochs": 3, "qlora": False},
        "distillation": {"enabled": False},
        "quantization": {"enabled": True, "quant_type": "nf4"},
    })
    # M4: Heuristic + FT + KD + Quant
    write_config(f"M4_{n_remove}", {
        "experiment_id": f"M4_{n_remove}",
        "description": f"Heuristic pruning ({n_remove} removed) + FT + KD + INT4",
        "base_model": "CohereForAI/aya-expanse-8b",
        "seed": 42,
        "pruning": {"method": "heuristic", "target_layers": target, "val_size": 200},
        "finetuning": {"enabled": True, "epochs": 3, "qlora": False},
        "distillation": {"enabled": True},
        "quantization": {"enabled": True, "quant_type": "nf4"},
    })

# ── Group 3: IFR-guided pruning ─────────────────────────────────────────────
print("Group 3: IFR-guided pruning")
for n_remove, target in PRUNE_TARGETS.items():
    # I1: IFR + FT
    write_config(f"I1_{n_remove}", {
        "experiment_id": f"I1_{n_remove}",
        "description": f"IFR pruning ({n_remove} removed -> {target} layers) + fine-tuning",
        "base_model": "CohereForAI/aya-expanse-8b",
        "seed": 42,
        "pruning": {"method": "ifr", "n_remove": n_remove,
                    "scores_path": "experiments/results/ifr_scores.json"},
        "finetuning": {"enabled": True, "epochs": 3, "qlora": False},
        "distillation": {"enabled": False},
        "quantization": {"enabled": False},
    })
    # I2: IFR + FT + KD
    write_config(f"I2_{n_remove}", {
        "experiment_id": f"I2_{n_remove}",
        "description": f"IFR pruning ({n_remove} removed) + FT + KD",
        "base_model": "CohereForAI/aya-expanse-8b",
        "seed": 42,
        "pruning": {"method": "ifr", "n_remove": n_remove,
                    "scores_path": "experiments/results/ifr_scores.json"},
        "finetuning": {"enabled": True, "epochs": 3, "qlora": False},
        "distillation": {"enabled": True},
        "quantization": {"enabled": False},
    })
    # I3: IFR + FT + Quant
    write_config(f"I3_{n_remove}", {
        "experiment_id": f"I3_{n_remove}",
        "description": f"IFR pruning ({n_remove} removed) + FT + INT4",
        "base_model": "CohereForAI/aya-expanse-8b",
        "seed": 42,
        "pruning": {"method": "ifr", "n_remove": n_remove,
                    "scores_path": "experiments/results/ifr_scores.json"},
        "finetuning": {"enabled": True, "epochs": 3, "qlora": False},
        "distillation": {"enabled": False},
        "quantization": {"enabled": True, "quant_type": "nf4"},
    })
    # I4: IFR + FT + KD + Quant
    write_config(f"I4_{n_remove}", {
        "experiment_id": f"I4_{n_remove}",
        "description": f"IFR pruning ({n_remove} removed) + FT + KD + INT4",
        "base_model": "CohereForAI/aya-expanse-8b",
        "seed": 42,
        "pruning": {"method": "ifr", "n_remove": n_remove,
                    "scores_path": "experiments/results/ifr_scores.json"},
        "finetuning": {"enabled": True, "epochs": 3, "qlora": False},
        "distillation": {"enabled": True},
        "quantization": {"enabled": True, "quant_type": "nf4"},
    })

# I5: IFR threshold-based + FT + Quant
write_config("I5_threshold", {
    "experiment_id": "I5_threshold",
    "description": "IFR threshold-based pruning + FT + INT4",
    "base_model": "CohereForAI/aya-expanse-8b",
    "seed": 42,
    "pruning": {"method": "ifr", "threshold_factor": 0.5,
                "scores_path": "experiments/results/ifr_scores.json"},
    "finetuning": {"enabled": True, "epochs": 3, "qlora": False},
    "distillation": {"enabled": False},
    "quantization": {"enabled": True, "quant_type": "nf4"},
})

# ── Group 4: LRP-guided pruning (placeholder) ───────────────────────────────
print("Group 4: LRP-guided pruning")
for n_remove, target in PRUNE_TARGETS.items():
    for variant, kd, quant in [("L1", False, False), ("L2", True, False),
                                ("L3", False, True), ("L4", True, True)]:
        write_config(f"{variant}_{n_remove}", {
            "experiment_id": f"{variant}_{n_remove}",
            "description": f"LRP pruning ({n_remove} removed) + FT{' + KD' if kd else ''}{' + INT4' if quant else ''}",
            "base_model": "CohereForAI/aya-expanse-8b",
            "seed": 42,
            "pruning": {"method": "lrp", "n_remove": n_remove},
            "finetuning": {"enabled": True, "epochs": 3, "qlora": False},
            "distillation": {"enabled": kd},
            "quantization": {"enabled": quant, "quant_type": "nf4"} if quant else {"enabled": False},
        })

print(f"\nDone! Configs written to {CONFIGS_DIR}")
