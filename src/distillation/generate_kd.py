"""Knowledge Distillation data generation using Aya Expanse 32B as teacher."""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.config import (
    KD_COMET_THRESHOLD,
    KD_DIR,
    SRC_LANG_NAME,
    TEACHER_MODEL,
    TGT_LANG_NAME,
    TRANSLATION_PROMPT,
)
from src.evaluation.metrics import compute_comet
from src.evaluation.translate import translate_with_vllm
from src.utils import load_env


def generate_kd_data(
    src_path: Path,
    ref_path: Path,
    output_dir: Path,
    teacher_model: str = TEACHER_MODEL,
    max_new_tokens: int = 256,
    tensor_parallel_size: int = 4,
    comet_threshold: float = KD_COMET_THRESHOLD,
    src_ext: str = "cs",
    tgt_ext: str = "de",
) -> Path:
    """Generate synthetic translations using teacher model, then filter by COMET.

    Args:
        src_path: Path to source sentences.
        ref_path: Path to reference translations (for COMET filtering).
        output_dir: Where to save KD data.
        teacher_model: Teacher model name/path.
        max_new_tokens: Max tokens for generation.
        tensor_parallel_size: vLLM tensor parallelism.
        comet_threshold: Minimum COMET score to keep a synthetic pair.

    Returns:
        Path to the output directory with filtered KD data.
    """
    load_env()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source data
    with open(src_path) as f:
        sources = f.read().splitlines()
    with open(ref_path) as f:
        references = f.read().splitlines()

    # Create prompts
    prompts = [
        TRANSLATION_PROMPT.format(
            src_lang=SRC_LANG_NAME,
            tgt_lang=TGT_LANG_NAME,
            source=src,
        )
        for src in sources
    ]

    # Generate translations with teacher
    print(f"Generating translations with {teacher_model} ...")
    hypotheses = translate_with_vllm(
        teacher_model,
        prompts,
        max_new_tokens=max_new_tokens,
        tensor_parallel_size=tensor_parallel_size,
    )

    # Save all generated translations before filtering
    raw_path = output_dir / "kd_raw.jsonl"
    with open(raw_path, "w") as f:
        for src, hyp, ref in zip(sources, hypotheses, references):
            f.write(json.dumps({"source": src, "hypothesis": hyp, "reference": ref}) + "\n")
    print(f"Saved {len(hypotheses)} raw KD pairs to {raw_path}")

    # Compute COMET scores for filtering
    print("Computing COMET scores for filtering ...")
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)

    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]
    output = comet_model.predict(data, batch_size=32, gpus=1)
    scores = output.scores

    # Filter by threshold, stripping embedded newlines from teacher outputs
    filtered_src = []
    filtered_tgt = []
    for src, hyp, score in zip(sources, hypotheses, scores):
        if score >= comet_threshold:
            filtered_src.append(src.replace("\n", " "))
            filtered_tgt.append(hyp.replace("\n", " "))

    print(f"Filtered: {len(filtered_src)}/{len(sources)} pairs "
          f"above COMET threshold {comet_threshold}")

    # Save filtered pairs
    src_out = output_dir / f"kd.{src_ext}"
    tgt_out = output_dir / f"kd.{tgt_ext}"
    with open(src_out, "w") as f:
        f.write("\n".join(filtered_src) + "\n")
    with open(tgt_out, "w") as f:
        f.write("\n".join(filtered_tgt) + "\n")

    print(f"Saved filtered KD data to {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate KD data from teacher model")
    parser.add_argument("--src", type=Path, required=True, help="Source sentences")
    parser.add_argument("--ref", type=Path, required=True, help="Reference translations")
    parser.add_argument("--output-dir", type=Path, default=KD_DIR)
    parser.add_argument("--teacher", default=TEACHER_MODEL)
    parser.add_argument("--tp-size", type=int, default=4, help="Tensor parallel GPUs")
    parser.add_argument("--comet-threshold", type=float, default=KD_COMET_THRESHOLD)
    parser.add_argument("--src-ext", default="cs", help="Source file extension")
    parser.add_argument("--tgt-ext", default="de", help="Target file extension")
    args = parser.parse_args()

    generate_kd_data(
        src_path=args.src,
        ref_path=args.ref,
        output_dir=args.output_dir,
        teacher_model=args.teacher,
        tensor_parallel_size=args.tp_size,
        comet_threshold=args.comet_threshold,
        src_ext=args.src_ext,
        tgt_ext=args.tgt_ext,
    )


if __name__ == "__main__":
    main()
