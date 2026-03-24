#!/usr/bin/env python3
"""Create all English-Spanish experiment configs."""
import yaml
from pathlib import Path

configs_dir = Path("experiments/configs")

# IFR-guided en-es
for n in [8, 12, 16]:
    cfg = {
        "experiment_id": f"I1_{n}_enes",
        "description": f"IFR pruning ({n} removed) + FT (English-Spanish)",
        "base_model": "CohereForAI/aya-expanse-8b",
        "seed": 42,
        "lang_pair": "en-es",
        "pruning": {
            "method": "ifr",
            "n_remove": n,
            "scores_path": "experiments/results/ifr_scores_enes.json",
        },
        "finetuning": {"enabled": True, "epochs": 3, "qlora": False},
        "distillation": {"enabled": False},
        "quantization": {"enabled": False},
    }
    path = configs_dir / f"I1_{n}_enes.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Created {path.name}")

# Iterative en-es
for n in [8, 12, 16]:
    cfg = {
        "experiment_id": f"M1_{n}_enes",
        "description": f"Iterative pruning ({n} removed) + FT (English-Spanish)",
        "base_model": "CohereForAI/aya-expanse-8b",
        "seed": 42,
        "lang_pair": "en-es",
        "pruning": {
            "method": "heuristic",
            "target_layers": 32 - n,
            "val_size": 200,
        },
        "finetuning": {"enabled": True, "epochs": 3, "qlora": False},
        "distillation": {"enabled": False},
        "quantization": {"enabled": False},
    }
    path = configs_dir / f"M1_{n}_enes.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Created {path.name}")

print("Done: 8 en-es configs created")
