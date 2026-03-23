# Interpretability-Guided Layer Pruning for Multilingual Machine Translation

## Project Overview

This project investigates whether interpretability techniques (Information Flow Routes / IFR) can identify which layers of **Aya Expanse 8B** (32 layers, Cohere architecture) contribute least to **Czech → German** translation quality, enabling more principled layer pruning than existing heuristic approaches (Moslem et al., WMT 2025).

**Research question:** Can IFR-guided pruning match or outperform iterative heuristic pruning, while being orders of magnitude cheaper to compute?

- **Model:** Aya Expanse 8B (32 layers, 8B parameters)
- **Language pair:** Czech → German (CES-DEU)
- **Dataset:** News Commentary v18 (100K train, 500 test)
- **Primary metric:** COMET (wmt22-comet-da)
- **Secondary metrics:** chrF++, BLEU, model size (MB), inference speed (tok/s)

---

## Current Results (March 2026)

**36 of 41 experiments completed.** 5 remaining (M2_12, M2_16, M4_8, M4_12, M4_16 — heuristic + KD variants). LRP experiments (L*) were dropped — method not implemented.

### Key Findings

**1. IFR-guided pruning matches heuristic quality at a fraction of the cost.**

At 8 layers removed (25% compression), IFR-guided pruning matches or slightly outperforms the iterative heuristic approach from Moslem et al., while requiring only a single forward pass per example instead of O(n×k) evaluations:

| Method | Layers | COMET | chrF++ | BLEU | Pruning Cost |
|--------|--------|-------|--------|------|-------------|
| Baseline (B0) | 32 | 0.832 | 49.26 | 15.92 | — |
| **IFR + FT (I1_8)** | **24** | **0.860** | **49.88** | **21.51** | **~30 min (200 examples)** |
| Heuristic + FT (M1_8) | 24 | 0.855 | 49.17 | 20.05 | ~5 hours |
| IFR + FT (I1_12) | 20 | 0.828 | 47.15 | 18.32 | ~30 min |
| Heuristic + FT (M1_12) | 20 | 0.819 | 46.51 | 17.28 | ~8 hours |
| IFR + FT (I1_16) | 16 | 0.739 | 42.45 | 13.74 | ~30 min |
| Heuristic + FT (M1_16) | 16 | 0.764 | 43.12 | 14.52 | ~12 hours |

IFR outperforms heuristic at 8 and 12 layers removed. At aggressive pruning (16 layers, 50%), heuristic is more robust — likely because its iterative approach captures local compensation effects that IFR's global scoring misses.

**2. Pruned + fine-tuned models beat the unpruned baseline.**

All models with 8 layers removed and LoRA fine-tuning outperform the unpruned baseline (B0: COMET 0.832). This is consistent with Moslem et al.'s findings. Fine-tuning on the translation task is the primary quality driver.

**3. Quantization preserves quality but doesn't speed up inference on high-end GPUs.**

| Config | COMET | Size (MB) | Speed (tok/s) | Speedup |
|--------|-------|-----------|---------------|---------|
| I1_8 (fp16) | 0.860 | 11,984 | 70.2 | 1.16x |
| I3_8 (INT4) | 0.857 | 4,496 | 45.9 | 0.76x |
| I3_8_int8 (INT8) | 0.860 | 6,992 | 23.2 | 0.38x |

INT4 reduces size by 62% with negligible quality loss (0.003 COMET). INT8 preserves quality even better. However, BitsAndBytes quantization introduces dequantization overhead that actually **slows down** inference on H200 GPUs. Quantization's value is **memory reduction for deployment on constrained hardware**, not throughput.

**4. Knowledge distillation from Aya 32B does not improve results.**

| Without KD | COMET | With KD | COMET | Δ |
|-----------|-------|---------|-------|----|
| I1_8 | 0.860 | I2_8 | 0.847 | -0.013 |
| I1_12 | 0.828 | I2_12 | 0.820 | -0.008 |
| M1_8 | 0.855 | M2_8 | 0.838 | -0.017 |

KD consistently hurts. The Aya 32B teacher translations, even after COMET filtering, may introduce noise that conflicts with the authentic training data.

**5. Pruning delivers real inference speedup; 16 layers removed achieves 1.78x.**

| Layers Removed | Remaining | Speed (tok/s) | Speedup | Best COMET |
|---------------|-----------|---------------|---------|------------|
| 0 (baseline) | 32 | 60.5 | 1.00x | 0.832 |
| 8 | 24 | 70.2 | 1.16x | 0.860 |
| 12 | 20 | 90.0 | 1.49x | 0.828 |
| 16 | 16 | 107.5 | 1.78x | 0.764 |

### Full Results Table

| Experiment | Layers | COMET | chrF++ | BLEU | Size (MB) | tok/s | ΔCOMET |
|-----------|--------|-------|--------|------|-----------|-------|--------|
| **Baselines** | | | | | | | |
| B0 (baseline) | 32 | 0.832 | 49.26 | 15.92 | 15,312 | 60.5 | — |
| B1 (INT4) | 32 | 0.849 | 50.51 | 20.12 | 5,328 | 36.1 | +0.017 |
| B1_int8 | 32 | 0.841 | 50.46 | 18.36 | 8,656 | 18.0 | +0.009 |
| **IFR + FT** | | | | | | | |
| I1_8 | 24 | 0.860 | 49.88 | 21.51 | 11,984 | 70.2 | +0.028 |
| I1_12 | 20 | 0.828 | 47.15 | 18.32 | 10,320 | 90.0 | -0.004 |
| I1_16 | 16 | 0.739 | 42.45 | 13.74 | 8,656 | 107.5 | -0.093 |
| **IFR + FT + KD** | | | | | | | |
| I2_8 | 24 | 0.847 | 48.67 | 20.64 | 11,984 | 78.4 | +0.015 |
| I2_12 | 20 | 0.820 | 45.85 | 17.92 | 10,320 | 92.2 | -0.012 |
| I2_16 | 16 | 0.724 | 40.05 | 12.20 | 8,656 | 108.0 | -0.108 |
| **IFR + FT + INT4** | | | | | | | |
| I3_8 | 24 | 0.857 | 49.21 | 20.60 | 4,496 | 45.9 | +0.025 |
| I3_12 | 20 | 0.826 | 46.62 | 18.18 | 4,080 | 54.0 | -0.006 |
| I3_16 | 16 | 0.724 | 41.96 | 12.77 | 3,664 | 63.4 | -0.108 |
| **IFR + FT + INT8** | | | | | | | |
| I3_8_int8 | 24 | 0.860 | 50.06 | 21.83 | 6,992 | 23.2 | +0.028 |
| I3_12_int8 | 20 | 0.828 | 47.13 | 18.56 | 6,160 | 27.5 | -0.005 |
| I3_16_int8 | 16 | 0.732 | 42.62 | 13.52 | 5,328 | 33.9 | -0.101 |
| **IFR + FT + KD + INT4** | | | | | | | |
| I4_8 | 24 | 0.841 | 47.80 | 19.79 | 4,496 | 46.6 | +0.009 |
| I4_12 | 20 | 0.816 | 45.53 | 17.27 | 4,080 | 55.0 | -0.016 |
| I4_16 | 16 | 0.720 | 40.53 | 12.33 | 3,664 | 66.5 | -0.113 |
| **IFR + FT + KD + INT8** | | | | | | | |
| I4_8_int8 | 24 | 0.847 | 48.55 | 20.56 | 6,992 | 22.9 | +0.014 |
| I4_12_int8 | 20 | 0.819 | 45.89 | 17.85 | 6,160 | 26.5 | -0.013 |
| I4_16_int8 | 16 | 0.724 | 40.36 | 11.94 | 5,328 | 33.4 | -0.108 |
| **IFR threshold** | | | | | | | |
| I5_threshold | 22 | 0.847 | 48.69 | 20.20 | 4,288 | 49.0 | +0.015 |
| I5_threshold_int8 | 22 | 0.850 | 49.00 | 20.21 | 6,576 | 24.1 | +0.018 |
| **Heuristic + FT** | | | | | | | |
| M1_8 | 24 | 0.855 | 49.17 | 20.05 | 11,984 | 71.8 | +0.023 |
| M1_12 | 20 | 0.819 | 46.51 | 17.28 | 10,320 | 88.6 | -0.013 |
| M1_16 | 16 | 0.764 | 43.12 | 14.52 | 8,656 | 107.5 | -0.069 |
| **Heuristic + FT + KD** | | | | | | | |
| M2_8 | 24 | 0.838 | 47.16 | 18.89 | 11,984 | 78.3 | +0.006 |
| **Heuristic + FT + INT4** | | | | | | | |
| M3_8 | 24 | 0.850 | 48.62 | 19.75 | 4,496 | 43.9 | +0.018 |
| M3_12 | 20 | 0.815 | 45.94 | 16.76 | 4,080 | 53.1 | -0.017 |
| M3_16 | 16 | 0.756 | 42.53 | 13.94 | 3,664 | 65.4 | -0.076 |
| **Heuristic + FT + INT8** | | | | | | | |
| M3_8_int8 | 24 | 0.853 | 49.19 | 20.14 | 6,992 | 23.1 | +0.021 |
| M3_12_int8 | 20 | 0.817 | 46.08 | 17.06 | 6,160 | 27.5 | -0.015 |
| M3_16_int8 | 16 | 0.761 | 43.19 | 14.83 | 5,328 | 33.0 | -0.071 |
| **Heuristic + FT + KD + INT8** | | | | | | | |
| M4_8_int8 | 24 | 0.835 | 47.00 | 18.86 | 6,992 | 22.5 | +0.002 |
| M4_12_int8 | 20 | 0.805 | 44.59 | 16.11 | 6,160 | 27.2 | -0.027 |
| M4_16_int8 | 16 | 0.739 | 40.77 | 12.95 | 5,328 | 33.7 | -0.093 |

### Comparison with Moslem et al. (WMT 2025)

Our heuristic replication (M1) and IFR results compared to the [original paper](https://aclanthology.org/2025.wmt-1.78/). Moslem uses the same dataset (News Commentary v18) and filtering pipeline but a different random split (seed 0 vs our seed 42), so absolute numbers differ slightly. Relative patterns are consistent:

| Pruning Level | Moslem COMET | Our IFR COMET | Our Heuristic COMET |
|--------------|-------------|---------------|---------------------|
| Baseline (32L) | 87.18 | 83.21 | — |
| 24 layers (-8) | 85.70 | **85.97** | 85.54 |
| 20 layers (-12) | 83.95 | 82.84 | 81.90 |
| 16 layers (-16) | 79.39 | 73.91 | 76.35 |

At moderate pruning (8–12 layers), IFR matches Moslem. At aggressive pruning (16 layers), heuristic is more robust.

### Still Running

- **M2_12, M2_16**: Heuristic + KD (12/16 layers removed)
- **M4_8, M4_12, M4_16**: Heuristic + KD + INT4

---

## Quick Start

```bash
# 1. Setup
uv venv --python 3.11 && uv sync

# 2. Pre-download models on login node (compute nodes have no internet)
source .venv/bin/activate
python -c "from huggingface_hub import snapshot_download; snapshot_download('CohereForAI/aya-expanse-8b')"
python -c "from comet import download_model; download_model('Unbabel/wmt22-comet-da')"

# 3. Run a single experiment
sbatch scripts/slurm/run_experiment.sh experiments/configs/I1_8.yaml

# 4. Aggregate results
python -m src.evaluation.aggregate_results

# 5. Generate plots
python scripts/plot_results.py
```

## Data Pipeline

News Commentary v18 CES-DEU filtering (replicating Moslem et al.):

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

## Repository Structure

```
├── data/
│   ├── raw/                    # News Commentary v18 download + fastText LID model
│   ├── filtered/               # Filtered + split: train.{cs,de}, test.{cs,de}
│   └── kd/                     # Synthetic KD data from Aya-32B teacher (98K pairs)
├── src/
│   ├── config.py               # Central constants (model names, thresholds, paths)
│   ├── utils.py                # Seed, env loading, device helpers
│   ├── run_experiment.py       # Master runner: reads YAML config, runs full pipeline
│   ├── data_prep/              # Download, filter, split News Commentary v18
│   ├── attribution/
│   │   ├── ifr.py              # IFR: HF hooks on residual stream, L1 proximity scores
│   │   └── score_layers.py     # CLI: score all layers, save ranking to JSON
│   ├── pruning/
│   │   ├── remove_layers.py    # Physical layer removal + layer_idx re-indexing
│   │   ├── heuristic.py        # Moslem iterative pruning (chrF++ guided)
│   │   └── guided.py           # IFR-guided: fixed-count or threshold-based selection
│   ├── finetuning/train.py     # LoRA fine-tuning via PEFT + TRL (r=16, α=32)
│   ├── distillation/
│   │   ├── generate_kd.py      # Aya-32B teacher translations via vLLM, COMET filtered
│   │   └── train_kd.py         # Fine-tune on merged authentic + KD data
│   ├── quantization/
│   │   └── quantize.py         # BitsAndBytes INT4/INT8 quantization
│   └── evaluation/
│       ├── translate.py        # Batch translation with stop-string hallucination guard
│       ├── metrics.py          # COMET, chrF++, BLEU, model size, inference speed
│       ├── run_eval.py         # CLI evaluation
│       └── aggregate_results.py # Collect all results into CSV
├── experiments/
│   ├── configs/                # 59 YAML configs (B*, M*, I* + INT8 variants)
│   └── results/                # Per-experiment: results.json + model artifacts
├── scripts/
│   ├── orchestrator.py         # Autonomous SLURM orchestrator with Claude Code
│   ├── plot_results.py         # Generate comparison plots
│   ├── validate_translation.py # Quick translation sanity check
│   ├── generate_configs.py     # Generate all experiment YAML configs
│   ├── smoke_test.py           # Validate model loading, IFR hooks, pruning
│   ├── submit_all.sh           # Submit all jobs with SLURM dependency chains
│   └── slurm/                  # SLURM batch scripts
├── paper/                      # LaTeX source (EMNLP 2023 style)
├── tests/                      # 170 unit tests (CPU, no GPU needed)
└── pyproject.toml              # UV/pip dependencies
```

---

## How It Works

### 1. IFR Attribution (`src/attribution/ifr.py`)

Custom implementation of Information Flow Routes (Ferrando & Voita, EMNLP 2024). Uses HuggingFace `register_forward_hook()` on the residual stream, computing proximity-based L1 contribution scores:

```
proximity(component, total) = max(-||total - component||₁ + ||total||₁, 0)
```

Layers with low importance scores across many translation examples are pruning candidates. Single forward pass per example — no gradient computation needed.

### 2. Layer Pruning (`src/pruning/`)

- **Heuristic (Moslem replication):** For each remaining layer (first/last protected), temporarily remove it, translate 200 sentences, measure chrF++. Remove the least-impactful layer. Repeat until target reached. O(n×k) evaluations.
- **IFR-guided:** Score layers once on 200 examples, remove the N lowest. O(n) cost.

After removal, `self_attn.layer_idx` is re-indexed to prevent KV cache IndexError.

### 3. Fine-tuning (`src/finetuning/train.py`)

LoRA (r=16, α=32) targeting all linear layers. 3 epochs, cosine schedule. Merges weights after training.

### 4. Knowledge Distillation (`src/distillation/`)

Aya 32B teacher generates synthetic translations via vLLM (1 GPU, fp16), filtered by COMET ≥ 0.7, merged with authentic training data. **Note:** KD did not improve results in our experiments.

### 5. Quantization (`src/quantization/quantize.py`)

BitsAndBytes INT4 (NF4, double quantization) and INT8. Reduces model size significantly but slows inference on high-VRAM GPUs due to dequantization overhead.

---

## Experimental Design

59 experiment configs across baselines, heuristic (Moslem replication), and IFR-guided variants. Each varies one factor at a time from the baseline.

| Group | Pruning | FT | KD | Quant | Configs |
|-------|---------|----|----|-------|---------|
| B0 | None | No | No | No | 1 |
| B1 | None | No | No | INT4/INT8 | 2 |
| M1 | Heuristic (8/12/16) | Yes | No | No | 3 |
| M2 | Heuristic (8/12/16) | Yes | Yes | No | 3 |
| M3 | Heuristic (8/12/16) | Yes | No | INT4/INT8 | 6 |
| M4 | Heuristic (8/12/16) | Yes | Yes | INT4/INT8 | 6 |
| I1 | IFR (8/12/16) | Yes | No | No | 3 |
| I2 | IFR (8/12/16) | Yes | Yes | No | 3 |
| I3 | IFR (8/12/16) | Yes | No | INT4/INT8 | 6 |
| I4 | IFR (8/12/16) | Yes | Yes | INT4/INT8 | 6 |
| I5 | IFR (threshold) | Yes | No | INT4/INT8 | 2 |

All experiments use seed 42, Aya Expanse 8B as base model, and 500 test sentences.

---

## Cluster Notes (BYU RC)

| Detail | Value |
|--------|-------|
| Account | `sdrich` |
| Primary GPU | `m13h` (H200 150GB, `--qos=gpu`) |
| Fallback GPU | `cs` (A100 80GB, `--qos=cs`) |
| Compute internet | **None** — all models pre-downloaded |
| Python | 3.11, managed via UV |
| Tests | `python -m pytest tests/ -v` (170 tests, CPU only) |

---

## References

- Dang et al. (2024). Aya Expanse. arXiv:2412.04261
- Ferrando & Voita (2024). Information Flow Routes. EMNLP 2024.
- Gaido et al. (2025). WMT 2025 Model Compression Shared Task.
- Moslem et al. (2025). [Iterative Layer Pruning for Efficient Translation Inference](https://aclanthology.org/2025.wmt-1.78/). WMT 2025.
- Vakilzadeh Hatefi et al. (2025). Attribution-guided Pruning. arXiv:2506.13727
