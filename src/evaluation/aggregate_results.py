"""Aggregate all experiment results into a single comparison table."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR


def collect_results(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Scan all experiment result directories and build a comparison table."""
    rows = []

    for result_file in sorted(results_dir.glob("*/results.json")):
        with open(result_file) as f:
            data = json.load(f)

        metrics = data.get("metrics", {})
        model_size = metrics.get("model_size", {})
        speed = metrics.get("inference_speed", {})

        row = {
            "experiment_id": data.get("experiment_id", result_file.parent.name),
            "description": data.get("description", ""),
            "num_layers": data.get("num_layers", ""),
            "comet": metrics.get("comet"),
            "chrf": metrics.get("chrf"),
            "bleu": metrics.get("bleu"),
            "total_params": model_size.get("total_params"),
            "size_mb": model_size.get("size_mb"),
            "tokens_per_sec": speed.get("tokens_per_second"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by group then by experiment
    if not df.empty:
        df["_sort_key"] = df["experiment_id"].str.extract(r"^([A-Z]+)")[0]
        df = df.sort_values(["_sort_key", "experiment_id"]).drop(columns=["_sort_key"])

    return df


def print_table(df: pd.DataFrame):
    """Print a formatted comparison table."""
    if df.empty:
        print("No results found.")
        return

    # Format numeric columns
    display_df = df.copy()
    for col in ["comet", "chrf", "bleu"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else ""
            )
    if "total_params" in display_df.columns:
        display_df["total_params"] = display_df["total_params"].apply(
            lambda x: f"{x/1e6:.0f}M" if pd.notna(x) else ""
        )
    if "size_mb" in display_df.columns:
        display_df["size_mb"] = display_df["size_mb"].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else ""
        )
    if "tokens_per_sec" in display_df.columns:
        display_df["tokens_per_sec"] = display_df["tokens_per_sec"].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else ""
        )

    print(display_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=None,
                        help="Save CSV to this path")
    args = parser.parse_args()

    df = collect_results(args.results_dir)
    print_table(df)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
    else:
        # Always save a default CSV
        csv_path = args.results_dir / "all_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
