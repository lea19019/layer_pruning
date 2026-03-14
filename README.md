# Interpretability-Guided Layer Pruning for Multilingual Machine Translation

## Project Overview

This project investigates whether interpretability techniques can identify which layers
of a multilingual LLM contribute least to translation quality, enabling more principled
layer pruning decisions than existing heuristic approaches.

The central research question is: **can interpretability-guided layer pruning combined
with quantization achieve translation quality comparable to quantization alone, while
producing a smaller model?**

The model under study is **Aya Expanse 8B** (Dang et al., 2024) — a multilingual LLM
with 32 transformer layers. The target language pair is **Czech → German (CES-DEU)**.
Translation quality is measured using **COMET** (primary) and **chrF++** (secondary).

---

## Background and Motivation

The WMT 2025 Model Compression shared task (Gaido et al., 2025) showed that:
- **Quantization alone** (e.g., INT4 via BitsAndBytes) reduces model size by up to 65%
  while nearly preserving baseline COMET scores (~55).
- **Heuristic layer pruning** (Moslem et al., 2025) iteratively removes the layer with
  the lowest impact on chrF++ at each step, then fine-tunes the pruned model. This
  achieved COMET ~39.9 in the shared task for Czech→German (down from 55.3 baseline),
  though the standalone paper reported stronger results — a discrepancy we aim to
  investigate by replicating the work.
- Combining heuristic pruning + quantization degraded quality further rather than helping.

Our hypothesis is that **interpretability-guided pruning** — using attribution scores to
identify unimportant layers rather than running expensive iterative quality evaluations —
will produce better pruning decisions and better final quality when combined with
quantization.

---

## Interpretability Methods

### Primary: Information Flow Routes (IFR)
Paper: Ferrando & Voita, EMNLP 2024
https://aclanthology.org/2024.emnlp-main.965.pdf

IFR builds an attribution graph over the transformer's residual stream. For a set of
reference translation examples, it traces how information flows from input tokens to
the final prediction, computing an importance score for each attention head and FFN
layer based on their contribution to the output. A single forward pass is sufficient
(~100x faster than activation patching). Layers with consistently low importance
scores across the reference set are candidates for pruning.

**How we use it:** Run IFR on a sample of Czech→German translation pairs from News
Commentary. Aggregate per-layer importance scores across examples. Use this ranking
to decide which layers to remove — either targeting fixed counts (8, 12, 16 layers)
or using a threshold-based approach to let the attribution scores determine how many
layers to prune.

### Secondary (time permitting): LRP Attribution Pruning
Paper: Vakilzadeh Hatefi et al., 2025
https://arxiv.org/pdf/2506.13727

Layer-wise Relevance Propagation (LRP) assigns relevance scores to model components
via a single forward-backward pass. When run on general-purpose or task-specific
reference samples, it identifies which layers contribute least to predictions. Unlike
IFR which operates at the information flow / circuit level, LRP operates at the
parameter level (individual weights), making it suitable for both structured layer
pruning and unstructured weight pruning.

**How we use it:** Run LRP on the same Czech→German reference samples. Aggregate
layer-level relevance scores. Use these scores to rank and remove layers, following
the same pruning targets as IFR for direct comparison.

---

## Datasets

### Training / Fine-tuning
- **News Commentary v18** (CES-DEU): ~250K parallel segments before filtering.
  After filtering expect ~200K usable segments.
  Following Moslem et al.: deduplicate, max 200 words per segment, length ratio ≤ 1.5x,
  language detection via fastText (threshold 0.9), semantic filtering via mUSE
  (threshold 0.7 cosine similarity).
  Use 100K training segments + 500 held-out test segments.

### Knowledge Distillation Data
- Generate synthetic CES-DEU translations using **Aya Expanse 32B** as teacher model.
  Filter synthetic pairs by COMET ≥ 70% before mixing with authentic data.
  Following Moslem et al. for consistency.

### Evaluation
- **In-domain:** 500-segment holdout from News Commentary (same split as Moslem).
- **External:** WMT 2025 shared task test set for CES-DEU if available, for
  cross-study comparison.

---

## Experimental Matrix

All experiments use Aya Expanse 8B (32 layers, ~16GB in FP16) as the base model.
Pruning targets: **8 layers removed (→24)**, **12 layers removed (→20)**,
**16 layers removed (→16)**.

### Group 1: Baselines
| ID | Description | Pruning | FT | KD | Quant |
|----|-------------|---------|----|----|-------|
| B0 | Original model | None | No | No | No |
| B1 | Quantization only | None | No | No | INT4 |

### Group 2: Moslem Replication (Heuristic Pruning)
Pruning method: iterative — remove the layer whose absence causes the smallest chrF++
drop on a small validation set. Repeat until target layer count is reached.

| ID | Pruning | FT | KD | Quant |
|----|---------|----|----|-------|
| M1 | Heuristic (8/12/16) | Yes | No | No |
| M2 | Heuristic (8/12/16) | Yes | Yes | No |
| M3 | Heuristic (8/12/16) | Yes | No | Yes (INT4) |
| M4 | Heuristic (8/12/16) | Yes | Yes | Yes (INT4) |

### Group 3: IFR-Guided Pruning (Primary)
Pruning method: run IFR on reference CES-DEU samples, rank layers by aggregated
importance score, remove the N least important layers.
Also includes one threshold-based variant where N is determined by the scores
themselves rather than a fixed target.

| ID | Pruning | FT | KD | Quant |
|----|---------|----|----|-------|
| I1 | IFR (8/12/16) | Yes | No | No |
| I2 | IFR (8/12/16) | Yes | Yes | No |
| I3 | IFR (8/12/16) | Yes | No | Yes (INT4) |
| I4 | IFR (8/12/16) | Yes | Yes | Yes (INT4) |
| I5 | IFR (threshold-based) | Yes | No | Yes (INT4) |

### Group 4: LRP-Guided Pruning (Secondary / Time Permitting)
| ID | Pruning | FT | KD | Quant |
|----|---------|----|----|-------|
| L1 | LRP (8/12/16) | Yes | No | No |
| L2 | LRP (8/12/16) | Yes | Yes | No |
| L3 | LRP (8/12/16) | Yes | No | Yes (INT4) |
| L4 | LRP (8/12/16) | Yes | Yes | Yes (INT4) |

---

## Key Research Questions

1. Does the Moslem et al. standalone paper replicate? Or do results align more
   with the weaker numbers reported in the WMT 2025 shared task evaluation?
2. Does IFR-guided pruning produce better translation quality than heuristic pruning
   at the same compression level?
3. Can IFR + quantization match or exceed quantization-alone (B1) while being smaller?
4. Does fine-tuning + KD + quantization compound positively or negatively?
5. Does a threshold-based pruning approach (I5) find a better compression point than
   fixed layer counts?

---

## Evaluation Metrics

- **COMET** (wmt22-comet-da): primary metric, neural MT quality estimation.
- **chrF++**: character-level F-score, used during iterative pruning decisions in the
  Moslem approach.
- **BLEU**: reported for completeness / cross-study comparison.
- **Model size**: number of parameters and disk size (GB) after pruning and/or
  quantization.
- **Inference speed**: tokens/second, measured with vLLM.

---

## Infrastructure

- **Base model:** `CohereForAI/aya-expanse-8b` from HuggingFace
- **Teacher model (KD):** `CohereForAI/aya-expanse-32b`
- **Hardware target:** NVIDIA A100 80GB (training), A40 48GB acceptable for inference
- **Frameworks:** HuggingFace Transformers, vLLM, BitsAndBytes (quantization)
- **IFR implementation:** see Ferrando & Voita's codebase / ALTI attribution method
- **LRP implementation:** https://github.com/erfanhatefi/SparC3

---

## Repository Structure (intended)
```
├── data/
│   ├── raw/              # Raw News Commentary downloads
│   ├── filtered/         # After filtering pipeline
│   └── kd/               # Synthetic KD data from Aya-32B teacher
├── src/
│   ├── data_prep/        # Filtering, splitting, KD data generation
│   ├── attribution/      # IFR and LRP layer importance scoring
│   ├── pruning/          # Layer removal logic (heuristic + guided)
│   ├── finetuning/       # Fine-tuning scripts
│   ├── distillation/     # Knowledge distillation training
│   ├── quantization/     # BitsAndBytes INT4 quantization
│   └── evaluation/       # COMET, chrF++, BLEU scoring
├── experiments/
│   ├── configs/          # Config files per experiment ID (B0, M1, I1, etc.)
│   └── results/          # Output tables and logs
├── notebooks/            # Exploratory analysis, attribution visualization
└── README.md
```

---

## References

- Dang et al. (2024). Aya Expanse. arXiv:2412.04261
- Ferrando & Voita (2024). Information Flow Routes. EMNLP 2024.
- Gaido et al. (2025). WMT 2025 Model Compression Shared Task.
- Kocmi et al. (2025). WMT25 General MT Shared Task.
- Moslem et al. (2025). Iterative Layer Pruning for Efficient Translation Inference. WMT 2025.
- Vakilzadeh Hatefi et al. (2025). Attribution-guided Pruning. arXiv:2506.13727
- Zhu et al. (2023). Survey on Model Compression for LLMs. arXiv:2308.07633