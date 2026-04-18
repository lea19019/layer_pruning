# Interpretability-Guided Layer Pruning for Machine Translation

This project prunes layers of **Aya-Expanse 8B** (a 32-layer Cohere
multilingual LLM) using **Information Flow Routes (IFR)**, an interpretability
method that scores each layer's contribution to the residual stream on a
translation task.  The pruned model is then recovered with LoRA
fine-tuning and knowledge distillation from Aya-Expanse 32B.  The main goal is
to match heuristic pruning (Moslem et al., WMT 2025) at ~10x lower pruning
cost, and to study the fp16 / GPTQ-4bit tradeoff on the resulting smaller
models.

Two language pairs are covered:

- English -> Spanish (`en-es`)
- Czech -> German (`ces-deu`)

## Released models

Three en-es models are published on HuggingFace (fp16):

| HF repo | Layers | Recipe | COMET | chrF++ | BLEU |
|---------|-------:|--------|------:|-------:|-----:|
| [`adrianMT56/aya-enes-B4`](https://huggingface.co/adrianMT56/aya-enes-B4) | 32 | Baseline: LoRA FT + KD (no pruning) | 0.8930 | 68.08 | 47.35 |
| [`adrianMT56/aya-enes-I2-8`](https://huggingface.co/adrianMT56/aya-enes-I2-8) | 24 | IFR pruning (8 removed) + LoRA FT + KD | 0.8880 | 67.13 | 46.02 |
| [`adrianMT56/aya-enes-I5-t05-kd`](https://huggingface.co/adrianMT56/aya-enes-I5-t05-kd) | 23 | IFR threshold=0.5 (9 removed) + LoRA FT + KD | 0.8863 | 66.85 | 45.47 |

All three are evaluated on the same held-out set of 500 News Commentary v18
en-es sentences.

## Repository layout

```
src/                      Pipeline modules (see "Code map" below)
scripts/                  CLI entry points + orchestration
scripts/slurm/            SLURM job scripts for the BYU RC cluster
experiments/configs/      YAML experiment configs (B0-B4, I1-I5, M1-M5, etc.)
tests/                    Pytest suite (157 tests, all run on CPU)
pyproject.toml            UV / pip dependencies
data.zip                  Provided separately (see "Data")
```

## Setup

The project uses [UV](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/adrianMT56/attention_lp.git
cd attention_lp
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Running inference on the released models (CPU)

A single script runs fp16 translation on CPU (or GPU if you have one):

```bash
# Translate one sentence
python scripts/inference.py \
    --model adrianMT56/aya-enes-I5-t05-kd \
    --text "The quick brown fox jumps over the lazy dog."

# Translate a file (one sentence per line)
python scripts/inference.py \
    --model adrianMT56/aya-enes-B4 \
    --input examples.en --output translations.es

# Czech -> German with the same script (only the prompt changes)
python scripts/inference.py \
    --model adrianMT56/aya-enes-B4 \
    --src-lang Czech --tgt-lang German \
    --text "Rychla hneda liska preskakuje liveho psa."

# Use GPU (if available) with fp16
python scripts/inference.py --model <id> --device cuda --dtype float16 --text "..."
```

CPU inference of the 22-23 layer pruned models takes ~30 s per sentence on a
modern laptop; fp16 on a 12 GB GPU is ~1 s per sentence.

## Converting a released model to GPTQ 4-bit

GPTQ quantization requires a GPU (gptqmodel does not support CPU quantization).

```bash
python scripts/quantize_to_gptq.py \
    --model adrianMT56/aya-enes-I5-t05-kd \
    --output-dir gptq_out/I5_t05_kd_enes
```

Calibration defaults to 128 lines from `data/filtered_en_es/test.en`; pass
`--calibration <path>` to use your own.  After quantization, the saved folder
can be loaded with `scripts/inference.py --model gptq_out/...` exactly like the
fp16 model.

## Data

`data.zip` (447 MB) is shipped separately. Unzip at the repo root:

```bash
unzip data.zip
```

Layout after unzip:

```
data/
  raw/                News Commentary v18 cs-de and en-es (downloaded)
  filtered/           cs-de after dedup + length + langid + semantic filtering (100k train + 500 test)
  filtered_en_es/     same pipeline, en-es
  kd/                 synthetic cs-de translations from Aya-Expanse 32B (COMET >= 0.7)
  kd_en_es/           synthetic en-es KD set
  DATA_CARD.md        filtering pipeline details, stats, splits
```

## Reproducing the training pipeline

All training was done on the BYU RC SLURM cluster. A single experiment is
driven by a YAML config:

```bash
sbatch scripts/slurm/run_experiment.sh experiments/configs/I2_8_enes.yaml
```

The three released models correspond to these configs:

- `experiments/configs/B4_enes.yaml`
- `experiments/configs/I2_8_enes.yaml`
- `experiments/configs/I5_t05_kd_enes.yaml`

The pipeline per experiment: `score -> prune -> fine-tune (+KD) -> (optional) quantize -> evaluate`.

To run the whole campaign (all configs, with SLURM dependency chains):

```bash
bash scripts/submit_all.sh
```

Results land in `experiments/results/<experiment_id>/results.json` and can be
aggregated with `python -m src.evaluation.aggregate_results`.

## Publishing the released models to HuggingFace

The three released models are uploaded from their local merged fp16 folders:

```bash
# Login once (prompts for your HF token)
huggingface-cli login

# Upload all three (dry-run first to verify)
python scripts/push_to_hf.py --dry-run
python scripts/push_to_hf.py
```

The script generates model cards from `experiments/results/*/results.json` and
the pruning info; re-running on an existing repo updates the card and weights.

## Code map

| Module | Role |
|--------|------|
| `src/config.py` | Model names, paths, hyperparameters |
| `src/data_prep/` | Download + filter News Commentary (dedup, length, langid, semantic) |
| `src/attribution/ifr.py` | IFR scoring via forward hooks on the residual stream |
| `src/attribution/score_layers.py` | CLI: score all 32 layers, save ranking |
| `src/pruning/heuristic.py` | Moslem et al. iterative pruning (chrF++-based) |
| `src/pruning/guided.py` | IFR-guided pruning (fixed-count or threshold) |
| `src/pruning/remove_layers.py` | Physical layer deletion + `layer_idx` re-indexing |
| `src/finetuning/train.py` | LoRA / QLoRA via PEFT + TRL |
| `src/distillation/generate_kd.py` | Teacher generation with vLLM (Aya-Expanse 32B) |
| `src/distillation/train_kd.py` | Fine-tuning on authentic + KD merged data |
| `src/quantization/quantize.py` | BitsAndBytes INT4/INT8 |
| `src/evaluation/translate.py` | Batched translation (HF or vLLM) |
| `src/evaluation/metrics.py` | COMET, chrF++, BLEU, size, speed |
| `src/evaluation/run_eval.py` | CLI: evaluate a model on a test set |
| `src/run_experiment.py` | Master runner reading a YAML config |
| `scripts/inference.py` | **CPU-friendly CLI for the released models** |
| `scripts/quantize_to_gptq.py` | **Convert any checkpoint to GPTQ 4-bit** |
| `scripts/push_to_hf.py` | Push the three released models to HF |

## Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

157 tests, all on CPU with mock models, no downloads.
