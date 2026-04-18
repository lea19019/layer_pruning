#!/usr/bin/env python3
"""List intermediate model checkpoints that can be deleted.

Keeps the FINAL model for each experiment (needed for vLLM speed benchmarking).
Only deletes earlier pipeline stages from completed experiments.

Usage:
    python scripts/cleanup_intermediates.py              # just print the list
    python scripts/cleanup_intermediates.py --delete      # actually delete
"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    results_dir = Path("experiments/results")
    total_freed = 0
    total_kept = 0

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        rf = exp_dir / "results.json"
        if not rf.exists():
            continue

        with open(rf) as f:
            cfg = json.load(f).get("config", {})

        has_quant = cfg.get("quantization", {}).get("enabled", False)
        has_ft = cfg.get("finetuning", {}).get("enabled", False)
        has_pruning = cfg.get("pruning", {}).get("method", "none") != "none"

        # Final model = last pipeline stage
        keep = None
        if has_quant:
            keep = "quantized"
        elif has_ft:
            keep = "finetuned"
        elif has_pruning:
            keep = "pruned_model"

        for sub in ["pruned_model", "finetuned", "quantized"]:
            p = exp_dir / sub
            if not p.exists():
                continue
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 ** 3)

            if sub == keep:
                total_kept += size
                print("KEEP   {:>6.1f} GB  {}".format(size, p))
            else:
                total_freed += size
                print("DELETE {:>6.1f} GB  {}".format(size, p))
                if args.delete:
                    shutil.rmtree(p)

    print()
    print("To free: {:.1f} GB".format(total_freed))
    print("Keeping: {:.1f} GB (final models for vLLM benchmark)".format(total_kept))
    if not args.delete:
        print("Run with --delete to actually remove files.")


if __name__ == "__main__":
    main()
