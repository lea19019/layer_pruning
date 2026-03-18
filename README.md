# Interpretability-Guided Layer Pruning for Multilingual Machine Translation

## Quick Start

```bash
# 1. Setup (already done if you cloned and ran setup)
uv venv --python 3.11 && uv sync

# 2. Pre-download models on login node (compute nodes have no internet)
source .venv/bin/activate
python -c "from huggingface_hub import snapshot_download; snapshot_download('CohereForAI/aya-expanse-8b')"
python -c "from comet import download_model; download_model('Unbabel/wmt22-comet-da')"

# 3. Run data preparation
sbatch scripts/slurm/data_prep.sh

# 4. Run IFR scoring (after data prep completes)
sbatch scripts/slurm/ifr_score.sh

# 5. Run a single experiment
sbatch scripts/slurm/run_experiment.sh experiments/configs/I1_8.yaml

# 6. Or submit everything with dependency chains
bash scripts/submit_all.sh

# 7. Aggregate results
python -m src.evaluation.aggregate_results
```

## Data Pipeline Status

News Commentary v18 CES-DEU filtering results:

| Step | Pairs | Removed |
|------|------:|--------:|
| Raw | 243,807 | — |
| Deduplication | 241,831 | 1,976 |
| Length filter (max 200w, ratio ≤ 1.5) | 225,680 | 16,151 |
| Language detection (fastText ≥ 0.9) | 214,532 | 11,148 |
| Semantic similarity (MiniLM ≥ 0.7) | 208,812 | 5,720 |
| **Train split** | **100,000** | — |
| **Test split** | **500** | — |

---

## Project Overview

This project investigates whether interpretability techniques can identify which layers
of a multilingual LLM contribute least to translation quality, enabling more principled
layer pruning decisions than existing heuristic approaches.

**Research question:** Can interpretability-guided layer pruning combined with
quantization achieve translation quality comparable to quantization alone, while
producing a smaller model?

- **Model:** Aya Expanse 8B (32 layers, Cohere architecture)
- **Language pair:** Czech → German
- **Primary metric:** COMET (wmt22-comet-da)
- **Secondary metrics:** chrF++, BLEU, model size, inference speed

---

## Repository Structure

```
├── data/
│   ├── raw/                    # News Commentary v18 download + fastText LID model
│   ├── filtered/               # Filtered + split: train.{cs,de}, test.{cs,de}
│   └── kd/                     # Synthetic KD data from Aya-32B teacher
├── src/
│   ├── config.py               # Central constants (model names, thresholds, paths)
│   ├── utils.py                # Seed, env loading, device helpers
│   ├── run_experiment.py       # Master runner: reads YAML config, runs full pipeline
│   ├── data_prep/
│   │   ├── download.py         # Download News Commentary v18 cs-de
│   │   ├── filter.py           # Dedup, length, langid (fastText), semantic (MiniLM)
│   │   └── split.py            # Train/test split
│   ├── attribution/
│   │   ├── ifr.py              # IFR implementation using HF hooks on residual stream
│   │   └── score_layers.py     # CLI: score all layers, save ranking to JSON
│   ├── pruning/
│   │   ├── remove_layers.py    # Physical layer removal + re-indexing
│   │   ├── heuristic.py        # Moslem iterative pruning (remove least-impactful layer)
│   │   └── guided.py           # IFR-guided: fixed-count or threshold-based selection
│   ├── finetuning/
│   │   └── train.py            # LoRA/QLoRA fine-tuning via PEFT + TRL
│   ├── distillation/
│   │   ├── generate_kd.py      # Generate translations with Aya-32B, filter by COMET
│   │   └── train_kd.py         # Fine-tune on merged authentic + KD data
│   ├── quantization/
│   │   └── quantize.py         # BitsAndBytes INT4 (NF4) quantization
│   └── evaluation/
│       ├── translate.py        # Batch translation (HF generate or vLLM)
│       ├── metrics.py          # COMET, chrF++, BLEU, model size, inference speed
│       ├── run_eval.py         # CLI: evaluate a model on test set
│       └── aggregate_results.py # Collect all experiment results into CSV table
├── experiments/
│   ├── configs/                # 39 YAML configs (B0-B1, M1-M4, I1-I5, L1-L4)
│   └── results/                # Per-experiment output dirs with results.json
├── scripts/
│   ├── setup_cluster.sh        # Create dirs, install deps
│   ├── generate_configs.py     # Generate all 39 experiment YAML configs
│   ├── smoke_test.py           # Validate model loading, IFR hooks, pruning
│   ├── submit_all.sh           # Submit all jobs with SLURM dependency chains
│   └── slurm/                  # Individual SLURM job scripts
│       ├── data_prep.sh
│       ├── ifr_score.sh
│       ├── generate_kd.sh
│       ├── run_experiment.sh
│       └── run_heuristic.sh
├── paper/                      # LaTeX source for the paper
├── notebooks/                  # Exploratory analysis
└── pyproject.toml              # UV/pip dependencies
```

---

## How It Works

### 1. IFR Attribution (`src/attribution/ifr.py`)

Custom implementation of Information Flow Routes (Ferrando & Voita, EMNLP 2024).
Uses HuggingFace `register_forward_hook()` to capture residual stream states at each
layer, then computes proximity-based L1 contribution scores:

```
proximity(component, total) = max(-||total - component||_1 + ||total||_1, 0)
```

For each layer, we measure how much the attention output and FFN output contribute to
the residual stream progression. Summing both gives a per-layer importance score.
Layers with low scores across many translation examples are pruning candidates.

### 2. Layer Pruning (`src/pruning/`)

Two approaches:

- **Heuristic (Moslem replication):** Replicates the approach from Moslem et al.
  (WMT 2025). For each remaining layer (excluding the first and last, which are
  protected), temporarily remove it, translate 200 validation sentences using
  `apply_chat_template` prompts, and measure chrF++. Remove the layer whose absence
  causes the smallest quality drop. Repeat until the target layer count is reached.
  Uses bfloat16 precision. O(n*k) evaluations where n=layers and k=layers to remove.

- **IFR-guided:** Run IFR once on 200 examples, rank layers by importance, remove the
  N least important. O(n) cost — single forward pass per example.

After removing layers, `self_attn.layer_idx` must be re-indexed so the KV cache
doesn't throw IndexError.

### 3. Fine-tuning (`src/finetuning/train.py`)

LoRA fine-tuning targeting all linear layers (q/k/v/o_proj, gate/up/down_proj).
r=16, alpha=32. 3 epochs, cosine schedule. Merges LoRA weights after training.
QLoRA variant loads model in 4-bit for memory efficiency.

### 4. Knowledge Distillation (`src/distillation/`)

Generates synthetic translations using Aya Expanse 32B as teacher, then filters by
COMET ≥ 0.7 and merges with authentic training data.

**vLLM dependency:** The 32B teacher model is too large for a single GPU, so KD
generation uses [vLLM](https://docs.vllm.ai/) — a high-throughput LLM inference engine
that supports tensor parallelism across multiple GPUs. vLLM is used **only** for this
step (`src/distillation/generate_kd.py`); all other translation (evaluation, heuristic
pruning) uses standard HuggingFace `model.generate()`.

> **BYU RC note:** vLLM's Triton backend JIT-compiles CUDA extensions at runtime,
> which requires Python development headers (`Python.h`). Since `python3-devel` is not
> installed on BYU RC compute nodes, the headers were extracted from the RPM into
> `~/python3.11-headers2/`. The KD SLURM script sets
> `C_INCLUDE_PATH=/home/vacl2/python3.11-headers2/usr/include/python3.11` to make this
> work.

### 5. Evaluation (`src/evaluation/`)

- **COMET** (wmt22-comet-da): primary quality metric
- **chrF++**: character n-gram F-score, used for pruning decisions
- **BLEU**: for cross-study comparison
- **Model size**: parameter count + disk size in MB
- **Inference speed**: tokens/second (warmup + timed generation)

Results aggregated via `python -m src.evaluation.aggregate_results` into a comparison CSV.

---

## Experimental Matrix

39 experiments across 4 groups. Each experiment is a YAML config in `experiments/configs/`.

### Group 1: Baselines
| ID | Description | Pruning | FT | KD | Quant |
|----|-------------|---------|----|----|-------|
| B0 | Original model | None | No | No | No |
| B1 | Quantization only | None | No | No | INT4 |

### Group 2: Moslem Replication (Heuristic Pruning)
| ID | Pruning | FT | KD | Quant |
|----|---------|----|----|-------|
| M1 | Heuristic (8/12/16) | Yes | No | No |
| M2 | Heuristic (8/12/16) | Yes | Yes | No |
| M3 | Heuristic (8/12/16) | Yes | No | INT4 |
| M4 | Heuristic (8/12/16) | Yes | Yes | INT4 |

### Group 3: IFR-Guided Pruning
| ID | Pruning | FT | KD | Quant |
|----|---------|----|----|-------|
| I1 | IFR (8/12/16) | Yes | No | No |
| I2 | IFR (8/12/16) | Yes | Yes | No |
| I3 | IFR (8/12/16) | Yes | No | INT4 |
| I4 | IFR (8/12/16) | Yes | Yes | INT4 |
| I5 | IFR (threshold) | Yes | No | INT4 |

### Group 4: LRP-Guided Pruning (time permitting)
| ID | Pruning | FT | KD | Quant |
|----|---------|----|----|-------|
| L1 | LRP (8/12/16) | Yes | No | No |
| L2 | LRP (8/12/16) | Yes | Yes | No |
| L3 | LRP (8/12/16) | Yes | No | INT4 |
| L4 | LRP (8/12/16) | Yes | Yes | INT4 |

---

## Cluster Notes (BYU RC)

| Detail | Value |
|--------|-------|
| Account | `sdrich` |
| Best GPU partition | `m13h` (H200, 150GB VRAM, `--qos=gpu`) |
| Alternative | `cs` (A100 80GB, `--qos=cs`) — often busy |
| Max walltime | 3 days (m13h), 7 days (cs) |
| Compute internet | **None** — pre-download everything on login node |
| Storage | Home dir: 2TB |
| HF cache | `~/.cache/huggingface/hub/` |

All SLURM scripts set `HF_HUB_OFFLINE=1` and use `#SBATCH --chdir=/home/vacl2/attention_lp`.

### Running individual experiments

```bash
# IFR-guided, 8 layers removed, with fine-tuning
sbatch scripts/slurm/run_experiment.sh experiments/configs/I1_8.yaml

# Heuristic pruning (use dedicated script — longer runtime)
sbatch scripts/slurm/run_heuristic.sh experiments/configs/M1_8.yaml

# Interactive GPU session for debugging
srun --partition=m13h --qos=gpu --account=sdrich --gres=gpu:h200:1 --mem=64G --time=00:30:00 bash
```

---

## Key Research Questions

1. Does Moslem et al. replicate? Or do results match the weaker WMT 2025 shared task numbers?
2. Does IFR-guided pruning produce better quality than heuristic pruning at the same compression?
3. Can IFR + quantization match quantization-alone (B1) while being smaller?
4. Does fine-tuning + KD + quantization compound positively or negatively?
5. Does threshold-based pruning (I5) find a better compression point than fixed targets?

---

## References

- Dang et al. (2024). Aya Expanse. arXiv:2412.04261
- Ferrando & Voita (2024). Information Flow Routes. EMNLP 2024.
- Gaido et al. (2025). WMT 2025 Model Compression Shared Task.
- Kocmi et al. (2025). WMT25 General MT Shared Task.
- Moslem et al. (2025). Iterative Layer Pruning for Efficient Translation Inference. WMT 2025.
- Vakilzadeh Hatefi et al. (2025). Attribution-guided Pruning. arXiv:2506.13727
- Zhu et al. (2023). Survey on Model Compression for LLMs. arXiv:2308.07633
