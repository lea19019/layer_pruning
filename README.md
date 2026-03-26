# Interpretability-Guided Layer Pruning for Multilingual Machine Translation

## Project Overview

This project investigates whether interpretability techniques (Information Flow Routes / IFR) can identify which layers of **Aya Expanse 8B** (32 layers, Cohere architecture) contribute least to translation quality, enabling more principled layer pruning than existing iterative approaches (Moslem et al., WMT 2025).

**Research question:** Can IFR-guided pruning match or outperform iterative empirical pruning, while being orders of magnitude cheaper to compute?

- **Model:** Aya Expanse 8B (32 layers, 8B parameters)
- **Language pairs:** Czech → German (CES-DEU), English → Spanish (ENG-SPA)
- **Dataset:** News Commentary v18 (100K train, 500 test per language pair)
- **Primary metric:** COMET (wmt22-comet-da)
- **Secondary metrics:** chrF++, BLEU, model size (MB), inference speed (tok/s)

---

## Current Status (March 2026)

**19 experiments completed and validated** (Czech-German). English-Spanish and KD experiments are pending re-run due to data alignment issues that have been fixed. See [Pending Experiments](#pending-experiments) below.

### Czech-German Results

#### Key Findings

**1. IFR-guided pruning matches iterative pruning quality at a fraction of the cost.**

| Method | Layers | COMET | chrF++ | BLEU | Pruning Cost |
|--------|--------|-------|--------|------|-------------|
| Baseline (B0) | 32 | 0.832 | 49.26 | 15.92 | — |
| **IFR + FT (I1_8)** | **24** | **0.860** | **49.88** | **21.51** | **~30 min** |
| Iterative + FT (M1_8) | 24 | 0.855 | 49.17 | 20.05 | ~5 hours |
| IFR + FT (I1_12) | 20 | 0.828 | 47.15 | 18.32 | ~30 min |
| Iterative + FT (M1_12) | 20 | 0.819 | 46.51 | 17.28 | ~8 hours |
| IFR + FT (I1_16) | 16 | 0.739 | 42.45 | 13.74 | ~30 min |
| Iterative + FT (M1_16) | 16 | 0.764 | 43.12 | 14.52 | ~12 hours |

IFR outperforms iterative at 8 and 12 layers removed. At 16 layers (50% compression), iterative is more robust.

**2. LoRA matches full fine-tuning.** B2 (LoRA, 0.870) vs B3 (full FT, 0.871) — validates LoRA for all experiments.

**3. Pruned + fine-tuned models beat the unpruned baseline** at 8 layers removed. Fine-tuning is the primary quality driver.

**4. INT8 quantization preserves quality, reduces size 42%, but slows inference** on high-VRAM GPUs due to BitsAndBytes dequantization overhead.

**5. Pruning delivers real inference speedup:** 1.16x at 8 layers, 1.49x at 12, 1.78x at 16.

#### Full Validated Results (Czech-German)

| Experiment | Layers | COMET | chrF++ | BLEU | Size (MB) | tok/s | ΔCOMET |
|-----------|--------|-------|--------|------|-----------|-------|--------|
| **Baselines** | | | | | | | |
| B0 (no FT) | 32 | 0.832 | 49.26 | 15.92 | 15,312 | 60.5 | — |
| B1_int8 (INT8 only) | 32 | 0.841 | 50.46 | 18.36 | 8,656 | 18.0 | +0.009 |
| B2 (LoRA, no pruning) | 32 | 0.870 | 52.00 | 23.91 | 15,312 | 58.7 | +0.038 |
| B3 (full FT, no pruning) | 32 | 0.871 | 52.42 | 24.70 | 15,312 | 59.2 | +0.039 |
| **IFR + FT** | | | | | | | |
| I1_8 | 24 | 0.860 | 49.88 | 21.51 | 11,984 | 70.2 | +0.028 |
| I1_12 | 20 | 0.828 | 47.15 | 18.32 | 10,320 | 90.0 | -0.004 |
| I1_16 | 16 | 0.739 | 42.45 | 13.74 | 8,656 | 107.5 | -0.093 |
| **IFR + FT + INT8** | | | | | | | |
| I3_8_int8 | 24 | 0.860 | 50.06 | 21.83 | 6,992 | 23.2 | +0.028 |
| I3_12_int8 | 20 | 0.828 | 47.13 | 18.56 | 6,160 | 27.5 | -0.005 |
| I3_16_int8 | 16 | 0.731 | 42.62 | 13.52 | 5,328 | 33.9 | -0.101 |
| **IFR threshold + INT8** | | | | | | | |
| I5_threshold_int8 | 22 | 0.850 | 49.00 | 20.21 | 6,576 | 24.1 | +0.018 |
| **Iterative + FT** | | | | | | | |
| M1_8 | 24 | 0.855 | 49.17 | 20.05 | 11,984 | 71.8 | +0.023 |
| M1_12 | 20 | 0.819 | 46.51 | 17.28 | 10,320 | 88.6 | -0.013 |
| M1_16 | 16 | 0.764 | 43.12 | 14.52 | 8,656 | 107.5 | -0.069 |
| **Iterative + FT + INT8** | | | | | | | |
| M3_8_int8 | 24 | 0.853 | 49.19 | 20.14 | 6,992 | 23.1 | +0.021 |
| M3_12_int8 | 20 | 0.817 | 46.08 | 17.06 | 6,160 | 27.5 | -0.015 |
| M3_16_int8 | 16 | 0.761 | 43.19 | 14.83 | 5,328 | 33.0 | -0.071 |

#### English-Spanish Baselines (validated)

| Experiment | Layers | COMET | chrF++ | BLEU |
|-----------|--------|-------|--------|------|
| B0_enes (no FT) | 32 | 0.582 | 42.88 | 9.61 |
| B1_int8_enes (INT8 only) | 32 | 0.602 | 44.86 | 10.90 |

#### Comparison with Moslem et al. (WMT 2025)

Compared to the [original paper](https://aclanthology.org/2025.wmt-1.78/). Same dataset and filtering pipeline, different random split (seed 0 vs 42).

| Pruning Level | Moslem COMET | Our IFR COMET | Our Iterative COMET |
|--------------|-------------|---------------|---------------------|
| Baseline (32L) | 87.18 | 83.21 | — |
| 24 layers (-8) | 85.70 | **85.97** | 85.54 |
| 20 layers (-12) | 83.95 | 82.84 | 81.90 |
| 16 layers (-16) | 79.39 | 73.91 | 76.35 |

Note: Moslem's baseline (87.18) benefits from instruction-tuning alignment. Our fine-tuned baselines B2/B3 (87.04/87.08) match this.

#### Layer Overlap: IFR vs Iterative

IFR removes a contiguous block of middle layers (8–18). Iterative pruning is more scattered, reaching into higher layers (23, 26, 28).

| Layers Removed | IFR (sorted) | Iterative (sorted) | Overlap | Jaccard |
|---------------|-------------|-------------------|---------|---------|
| 8 | 8,10,11,12,13,14,15,16 | 7,9,12,16,17,18,23,26 | 2/8 | 0.14 |
| 12 | 7–18 | 7,9,10,12,15–18,22,23,26,28 | 8/12 | 0.50 |
| 16 | 6–18,20,22,23 | 7–18,22,23,26,28 | 14/16 | 0.78 |

Layer removal data is saved per experiment: `pruning_info.json` (IFR) and `pruning_log.json` (iterative).

---

## Pending Experiments

The following experiments need to be submitted. All code and configs are ready. Data alignment issues have been fixed.

### Currently Running

- **En-es KD data generation** (job 11081849) — Aya 32B teacher translating 100K English sentences to Spanish via vLLM. Output: `data/kd_en_es/`.

### Czech-German KD experiments (12)

Previous results were invalid due to misaligned merged training data (embedded newlines in teacher output caused 2,262 extra German lines). The source KD data (`data/kd/kd.cs`, `data/kd/kd.de`) is now clean. The stale merged file (`data/filtered/train_kd.de`) has been deleted and will be regenerated correctly on next run.

Submit with: `sbatch scripts/slurm/batch_kd_rerun.sh`

Experiments: I2_8, I2_12, I2_16, I4_8_int8, I4_12_int8, I4_16_int8, M2_8, M2_12, M2_16, M4_8_int8, M4_12_int8, M4_16_int8

### English-Spanish core experiments (10)

Previous results were invalid due to: (1) Unicode Line Separator (U+2028) in 2 English training sentences causing 100,002 vs 100,000 line misalignment, now fixed; (2) missing English/Spanish stop strings in translation post-processing, now added.

Submit with: `sbatch scripts/slurm/batch_enes.sh`

Experiments: B2_enes, B3_enes, I1_8_enes, I1_12_enes, I1_16_enes, M1_8_enes, M1_12_enes, M1_16_enes (B0_enes and B1_int8_enes are already valid since they don't use training data)

### English-Spanish non-KD experiments (7)

Submit with: `sbatch scripts/slurm/batch_enes_nokd.sh`

Experiments: I3_8_int8_enes, I3_12_int8_enes, I3_16_int8_enes, I5_threshold_int8_enes, M3_8_int8_enes, M3_12_int8_enes, M3_16_int8_enes

### English-Spanish KD experiments (12)

Requires en-es KD data generation to complete first.

Submit with: `sbatch --dependency=afterok:<kd_job_id> scripts/slurm/batch_enes_kd.sh`

Experiments: I2_8_enes, I2_12_enes, I2_16_enes, I4_8_int8_enes, I4_12_int8_enes, I4_16_int8_enes, M2_8_enes, M2_12_enes, M2_16_enes, M4_8_int8_enes, M4_12_int8_enes, M4_16_int8_enes

---

## Data

### Czech-German (News Commentary v18)

| Step | Pairs | Removed |
|------|------:|--------:|
| Raw | 243,807 | — |
| Deduplication | 241,831 | 1,976 |
| Length filter (max 200w, ratio ≤ 1.5) | 225,680 | 16,151 |
| Language detection (fastText ≥ 0.9) | 214,532 | 11,148 |
| Semantic similarity (MiniLM ≥ 0.7) | 208,812 | 5,720 |
| **Train** | **100,000** | — |
| **Test** | **500** | — |

### English-Spanish (News Commentary v18)

| Step | Pairs | Removed |
|------|------:|--------:|
| Raw | 513,608 | — |
| Deduplication | 498,569 | 15,039 |
| Length filter | 464,721 | 33,848 |
| Language detection | 374,216 | 90,505 |
| Semantic similarity | ~350,000 | ~24,000 |
| **Train** | **100,000** | — |
| **Test** | **500** | — |

---

## Experimental Design

Each language pair runs the same set of experiments. Each varies one factor at a time from the baseline.

### Experiment Matrix (per language pair)

| Group | Pruning | FT Method | KD | Quant | Configs |
|-------|---------|-----------|-----|-------|---------|
| B0 | None | None | No | No | 1 |
| B1 | None | None | No | INT8 | 1 |
| B2 | None | LoRA | No | No | 1 |
| B3 | None | Full FT | No | No | 1 |
| M1 | Iterative (8/12/16) | LoRA | No | No | 3 |
| M2 | Iterative (8/12/16) | LoRA | Yes | No | 3 |
| M3 | Iterative (8/12/16) | LoRA | No | INT8 | 3 |
| M4 | Iterative (8/12/16) | LoRA | Yes | INT8 | 3 |
| I1 | IFR (8/12/16) | LoRA | No | No | 3 |
| I2 | IFR (8/12/16) | LoRA | Yes | No | 3 |
| I3 | IFR (8/12/16) | LoRA | No | INT8 | 3 |
| I4 | IFR (8/12/16) | LoRA | Yes | INT8 | 3 |
| I5 | IFR (threshold) | LoRA | No | INT8 | 1 |

**Total: 29 experiments × 2 language pairs = 58 experiments**

### Controlled Variables

| Variable | Value |
|----------|-------|
| Base model | Aya Expanse 8B (32 layers, 8B params) |
| Random seed | 42 |
| Training data | 100,000 sentence pairs per language pair |
| Test data | 500 sentence pairs per language pair |
| Max generation tokens | 256 |
| Evaluation metrics | COMET (wmt22-comet-da), chrF++, BLEU, model size, inference speed |

### Manipulated Variables

| Variable | Levels |
|----------|--------|
| Pruning method | None, Iterative empirical (Moslem et al.), IFR-guided, IFR threshold |
| Layers removed | 0, 8, 12, 16 (or auto via threshold) |
| Recovery fine-tuning | None, LoRA (r=16, α=32, 3 epochs), Full FT (1 epoch) |
| Knowledge distillation | No, Yes (Aya 32B teacher, COMET ≥ 0.7 filter) |
| Quantization | None, INT8 |
| Language pair | Czech→German, English→Spanish |

### Fine-tuning Hyperparameters

| Parameter | LoRA | Full FT |
|-----------|------|---------|
| Trainable parameters | ~26M (0.5%) | All (~8B, 100%) |
| Rank / Alpha | 16 / 32 | — |
| Target modules | q/k/v/o_proj, gate/up/down_proj | All |
| Epochs | 3 | 1 |
| Learning rate | 2e-5 | 2e-5 |
| Batch size | 8 | 8 |
| Gradient accumulation | 8 | 8 |
| Scheduler | Cosine, 5% warmup | Cosine, 5% warmup |
| Precision | fp16 | bf16 |

---

## Quick Start

```bash
# 1. Setup
uv venv --python 3.11 && uv sync

# 2. Pre-download models (login node only)
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

---

## Repository Structure

```
├── data/
│   ├── raw/                    # News Commentary v18 downloads + fastText LID model
│   ├── filtered/               # cs-de: train.{cs,de}, test.{cs,de}
│   ├── filtered_en_es/         # en-es: train.{en,es}, test.{en,es}
│   ├── kd/                     # cs-de KD data from Aya-32B (98K pairs)
│   └── kd_en_es/               # en-es KD data from Aya-32B
├── src/
│   ├── config.py               # Central constants
│   ├── utils.py                # Seed, env loading, device helpers
│   ├── run_experiment.py       # Master runner (language-aware, reads lang_pair from YAML)
│   ├── data_prep/              # Download, filter, split
│   ├── attribution/
│   │   ├── ifr.py              # IFR: HF hooks on residual stream, L1 proximity scores
│   │   └── score_layers.py     # CLI: score all layers (supports --src-lang-name/--tgt-lang-name)
│   ├── pruning/
│   │   ├── remove_layers.py    # Physical layer removal + layer_idx re-indexing
│   │   ├── heuristic.py        # Moslem iterative pruning (chrF++ guided)
│   │   └── guided.py           # IFR-guided: fixed-count or threshold-based
│   ├── finetuning/train.py     # LoRA or full FT via PEFT + TRL
│   ├── distillation/
│   │   ├── generate_kd.py      # Aya-32B teacher via vLLM, COMET filtered (language-aware)
│   │   └── train_kd.py         # Fine-tune on merged authentic + KD data (language-aware)
│   ├── quantization/
│   │   └── quantize.py         # BitsAndBytes INT8 quantization
│   └── evaluation/
│       ├── translate.py        # Batch translation with multi-language stop strings
│       ├── metrics.py          # COMET, chrF++, BLEU, model size, inference speed
│       ├── run_eval.py         # CLI evaluation
│       └── aggregate_results.py # Collect all results into CSV
├── experiments/
│   ├── configs/                # YAML configs (cs-de + en-es variants)
│   └── results/                # Per-experiment: results.json + model artifacts
├── scripts/
│   ├── plot_results.py         # Generate comparison plots
│   ├── prep_enes_data.py       # Filter and split en-es data
│   └── slurm/                  # SLURM batch scripts
├── tests/                      # 173 unit tests (CPU, no GPU needed)
├── FUTURE_WORK.md              # Plan: interpretability-guided mixed-precision quantization
└── pyproject.toml
```

---

## Cluster Notes (BYU RC)

| Detail | Value |
|--------|-------|
| Account | `sdrich` |
| Primary GPU | `m13h` (H200 150GB, `--qos=gpu`) |
| Fallback GPU | `cs` (A100 80GB, `--qos=cs`) |
| Compute internet | **None** — all models pre-downloaded |
| Python | 3.11, managed via UV |
| Tests | `python -m pytest tests/ -v` (173 tests, CPU only) |

---

## References

- Dang et al. (2024). Aya Expanse. arXiv:2412.04261
- Ferrando & Voita (2024). Information Flow Routes. EMNLP 2024.
- Gaido et al. (2025). WMT 2025 Model Compression Shared Task.
- Moslem et al. (2025). [Iterative Layer Pruning for Efficient Translation Inference](https://aclanthology.org/2025.wmt-1.78/). WMT 2025.
- Vakilzadeh Hatefi et al. (2025). Attribution-guided Pruning. arXiv:2506.13727
