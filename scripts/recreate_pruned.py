"""Recreate missing pruned_model/ directories from saved pruning_info.json.

For each experiment dir under experiments/results/, if pruning_info.json exists
and pruned_model/ does not, reload the base model, delete the saved layers,
and save to pruned_model/.

Usage:
    python scripts/recreate_pruned.py IP_12 IP_16 ...
    python scripts/recreate_pruned.py --all-missing
"""

import argparse
import json
from pathlib import Path

from src.config import BASE_MODEL
from src.pruning.remove_layers import load_and_prune, save_pruned_model

RESULTS_DIR = Path("experiments/results")


def needs_recreate(exp_id: str) -> tuple[bool, list[int] | None]:
    d = RESULTS_DIR / exp_id
    info = d / "pruning_info.json"
    if not info.exists():
        return False, None
    pruned = d / "pruned_model"
    if pruned.exists() and (pruned / "config.json").exists():
        return False, None
    with open(info) as f:
        meta = json.load(f)
    return True, meta.get("layers_removed", [])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiments", nargs="*")
    p.add_argument("--all-missing", action="store_true")
    args = p.parse_args()

    if args.all_missing:
        ids = [d.name for d in RESULTS_DIR.iterdir() if d.is_dir()]
    else:
        ids = args.experiments

    todo = []
    for eid in ids:
        do_it, layers = needs_recreate(eid)
        if do_it:
            todo.append((eid, layers))

    print(f"Will recreate {len(todo)} pruned models:")
    for eid, layers in todo:
        print(f"  {eid}: remove {len(layers)} layers {layers}")

    for eid, layers in todo:
        print(f"\n=== {eid}: removing {len(layers)} layers ===")
        out = RESULTS_DIR / eid / "pruned_model"
        try:
            model, tok = load_and_prune(BASE_MODEL, layers)
            save_pruned_model(model, tok, str(out))
            del model
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"FAILED {eid}: {e}")


if __name__ == "__main__":
    main()
