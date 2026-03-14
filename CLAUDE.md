# CLAUDE.md

## Project Summary

This project investigates whether interpretability techniques (Information Flow Routes / IFR) can identify which layers of Aya Expanse 8B (a 32-layer Cohere-architecture multilingual LLM) contribute least to Czech-to-German translation quality, enabling more principled layer pruning than heuristic approaches. The pipeline includes IFR attribution scoring, layer pruning (heuristic or IFR-guided), LoRA/QLoRA fine-tuning, optional knowledge distillation from Aya 32B, INT4 quantization, and evaluation via COMET/chrF++/BLEU. There are 39 experiment configurations across baselines, heuristic replication (Moslem et al.), and IFR-guided variants.

## Environment (BYU RC Cluster)

- **Compute nodes have NO internet access.** All models and data must be pre-downloaded on the login node.
- **Always set `HF_HUB_OFFLINE=1`** in SLURM scripts and any compute-node commands.
- **SLURM scripts must use `#SBATCH --chdir=/home/vacl2/attention_lp`** -- do not use `dirname $0` because SLURM copies scripts to a temp location.
- **Preferred partition:** `m13h` (H200 150GB VRAM, `--qos=gpu`). **Fallback:** `cs` (A100 80GB, `--qos=cs`).
- **Account:** `sdrich`
- **Home directory:** 2TB quota, no need for `/nobackup`.
- **Package manager:** UV. Virtual environment at `.venv/`. Activate with `source .venv/bin/activate`.
- **Python:** 3.11

## Running Tests

**Unit tests (CPU, login node — no GPU needed):**

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

157 tests covering all modules: config, utils, data_prep, attribution (with mock model), pruning (layer removal + re-indexing), evaluation (chrF++, BLEU, model size), finetuning, distillation, and experiment config validation. All run on CPU with mock models — no downloads required.

**GPU smoke test (compute node):**

```bash
srun --partition=m13h --qos=gpu --account=sdrich --gres=gpu:h200:1 --mem=64G --time=00:30:00 bash -c "cd ~/attention_lp && source .venv/bin/activate && export HF_HUB_OFFLINE=1 && python scripts/smoke_test.py"
```

Validates model loading, IFR hooks on real Aya 8B, and layer pruning with generation. Requires pre-downloaded model.

## Key Architecture Decisions

1. **IFR uses HF `register_forward_hook` directly** -- TransformerLens does not support the Cohere architecture. Hooks capture residual stream states (layer input, attention output, MLP output, layer output) to compute proximity-based L1 contribution scores.

2. **After layer removal, `self_attn.layer_idx` must be re-indexed** to sequential 0..N-1. Without this, the KV cache (sized by `num_hidden_layers`) throws `IndexError` from layers that kept their old index. See `src/pruning/remove_layers.py:44`.

3. **Left-padding is required for batched generation.** `tokenizer.padding_side = "left"` is set in `src/evaluation/translate.py`.

4. **COMET model should be cached globally** to avoid reloading per experiment. Pre-download with: `python -c "from comet import download_model; download_model('Unbabel/wmt22-comet-da')"`.

5. **Semantic filtering uses `sentence-transformers` (paraphrase-multilingual-MiniLM-L12-v2)**, not mUSE/TensorFlow.

6. **`torch.cuda.get_device_properties(0).total_memory`** -- use `.total_memory`, not `.total_mem` (PyTorch 2.9+).

## Common Commands

```bash
# Activate environment
source .venv/bin/activate

# Pre-download models (login node only, has internet)
python -c "from huggingface_hub import snapshot_download; snapshot_download('CohereForAI/aya-expanse-8b')"
python -c "from comet import download_model; download_model('Unbabel/wmt22-comet-da')"

# Submit a single experiment
sbatch scripts/slurm/run_experiment.sh experiments/configs/I1_8.yaml

# Submit a heuristic pruning experiment (longer runtime, dedicated script)
sbatch scripts/slurm/run_heuristic.sh experiments/configs/M1_8.yaml

# Submit all experiments with dependency chains
bash scripts/submit_all.sh

# Interactive GPU session for debugging
srun --partition=m13h --qos=gpu --account=sdrich --gres=gpu:h200:1 --mem=64G --time=00:30:00 bash

# Check job status
squeue -u $USER

# Aggregate all experiment results into CSV
python -m src.evaluation.aggregate_results

# Run smoke test on GPU node
python scripts/smoke_test.py
```

## Code Organization

| Module | Responsibility |
|--------|---------------|
| `src/config.py` | Central constants: model names, paths, thresholds, hyperparameters |
| `src/utils.py` | Seed setting, .env loading, device detection |
| `src/run_experiment.py` | Master runner: reads YAML config, executes prune -> finetune -> quantize -> evaluate pipeline |
| `src/data_prep/` | Download News Commentary v18 cs-de, filter (dedup, length, langid, semantic), split train/test |
| `src/attribution/ifr.py` | IFR implementation: forward hooks on residual stream, proximity-based L1 scoring per layer |
| `src/attribution/score_layers.py` | CLI to score all layers on dataset, save ranking to JSON |
| `src/pruning/remove_layers.py` | Physical layer deletion from `model.model.layers` + layer_idx re-indexing |
| `src/pruning/heuristic.py` | Moslem et al. replication: iterative remove-least-impactful-layer via chrF++ eval (O(n^2)) |
| `src/pruning/guided.py` | IFR-guided: fixed-count or threshold-based layer selection from pre-computed scores |
| `src/finetuning/train.py` | LoRA/QLoRA fine-tuning via PEFT + TRL (r=16, alpha=32, 3 epochs) |
| `src/distillation/generate_kd.py` | Generate synthetic translations from Aya 32B teacher, filter by COMET >= 0.7 |
| `src/distillation/train_kd.py` | Fine-tune on merged authentic + KD data |
| `src/quantization/quantize.py` | BitsAndBytes INT4 (NF4) quantization |
| `src/evaluation/translate.py` | Batch translation (HF generate with left-padding, or vLLM) |
| `src/evaluation/metrics.py` | COMET, chrF++, BLEU, model size, inference speed |
| `src/evaluation/run_eval.py` | CLI to evaluate a model on test set |
| `src/evaluation/aggregate_results.py` | Collect all `results.json` files into comparison CSV |
| `experiments/configs/` | 39 YAML experiment configs (B0-B1, M1-M4, I1-I5, L1-L4) |
| `scripts/submit_all.sh` | Submit all jobs with SLURM dependency chains |
| `scripts/smoke_test.py` | Validate model loading, IFR hooks, pruning on GPU |
| `scripts/generate_configs.py` | Generate all 39 YAML configs |

## Experiment Workflow

The correct order of operations:

1. **Pre-download models on login node** (has internet):
   ```bash
   source .venv/bin/activate
   python -c "from huggingface_hub import snapshot_download; snapshot_download('CohereForAI/aya-expanse-8b')"
   python -c "from comet import download_model; download_model('Unbabel/wmt22-comet-da')"
   ```

2. **Data prep** (already done: 100K train + 500 test in `data/filtered/`):
   ```bash
   sbatch scripts/slurm/data_prep.sh
   ```

3. **IFR scoring** (run on GPU, produces `experiments/results/ifr_scores.json`):
   ```bash
   sbatch scripts/slurm/ifr_score.sh
   ```

4. **Submit experiments** -- either individually or all at once:
   ```bash
   # Single experiment:
   sbatch scripts/slurm/run_experiment.sh experiments/configs/I1_8.yaml

   # All experiments with dependency chains:
   bash scripts/submit_all.sh
   ```

5. **Aggregate results** (after experiments complete):
   ```bash
   python -m src.evaluation.aggregate_results
   ```

Results are saved per-experiment in `experiments/results/<exp_id>/results.json`.

## SLURM Script Template

When creating new SLURM scripts, follow this pattern:

```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=HH:MM:SS
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/<name>_%j.out
#SBATCH --error=logs/<name>_%j.err

set -euo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

# commands here
```
