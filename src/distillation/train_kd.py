"""Fine-tuning with mixed authentic + KD data."""

from pathlib import Path

from src.config import FILTERED_DIR, KD_DIR
from src.finetuning.train import finetune


def merge_datasets(
    authentic_src: Path,
    authentic_tgt: Path,
    kd_src: Path,
    kd_tgt: Path,
    output_dir: Path,
    src_ext: str = "cs",
    tgt_ext: str = "de",
) -> tuple[Path, Path]:
    """Merge authentic and KD data into a single training set."""
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_src = output_dir / f"train_kd.{src_ext}"
    merged_tgt = output_dir / f"train_kd.{tgt_ext}"

    with open(authentic_src) as f:
        a_src = f.read().splitlines()
    with open(authentic_tgt) as f:
        a_tgt = f.read().splitlines()
    with open(kd_src) as f:
        k_src = f.read().splitlines()
    with open(kd_tgt) as f:
        k_tgt = f.read().splitlines()

    all_src = a_src + k_src
    all_tgt = a_tgt + k_tgt

    with open(merged_src, "w") as f:
        f.write("\n".join(all_src) + "\n")
    with open(merged_tgt, "w") as f:
        f.write("\n".join(all_tgt) + "\n")

    print(f"Merged: {len(a_src)} authentic + {len(k_src)} KD = {len(all_src)} total")
    return merged_src, merged_tgt


def finetune_with_kd(
    model_name_or_path: str,
    output_dir: Path,
    use_qlora: bool = False,
    data_dir: Path = FILTERED_DIR,
    kd_dir: Path = KD_DIR,
    src_ext: str = "cs",
    tgt_ext: str = "de",
):
    """Fine-tune using merged authentic + KD data."""
    # Merge datasets
    merged_src, merged_tgt = merge_datasets(
        authentic_src=data_dir / f"train.{src_ext}",
        authentic_tgt=data_dir / f"train.{tgt_ext}",
        kd_src=kd_dir / f"kd.{src_ext}",
        kd_tgt=kd_dir / f"kd.{tgt_ext}",
        output_dir=data_dir,
        src_ext=src_ext,
        tgt_ext=tgt_ext,
    )

    return finetune(
        model_name_or_path=model_name_or_path,
        train_src=merged_src,
        train_tgt=merged_tgt,
        output_dir=output_dir,
        use_qlora=use_qlora,
    )
