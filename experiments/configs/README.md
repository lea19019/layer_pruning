# Experiment configs

Two kinds of YAML files live here:

**`recipes/`** — five canonical templates that show the main pipeline patterns
(baseline / heuristic prune / IFR prune / full pipeline / custom dataset).
Start here when you want to run something new. Most of the time you can skip
the YAML entirely and use `attention-lp run ...` flags directly — see the
README at the repo root.

**`<ID>.yaml`** — historical configs for past experiments. Each file
corresponds to a run whose results live in `experiments/results/<ID>/` and
which is referenced by one or more SLURM batch scripts in `scripts/slurm/`.
Don't delete these — they're the reproducibility record.

## Naming scheme

`<prune>[_<layers>][_<suffix>]_[enes].yaml`

| Prefix | Meaning |
|--|--|
| `B0`–`B4` | Baselines (no pruning) |
| `M<n>_<L>` | Moslem iterative heuristic, keeps `L` layers |
| `I<n>_<L>` | IFR-guided, removes `L` layers (or threshold variant) |
| `I5_tNN` | IFR threshold-based, threshold = 0.NN |
| `IP_<L>`, `MP_<L>` | Prune-only variants (no fine-tuning) |
| `_int8` | 8-bit quantization |
| `_ft` | fine-tuning added |
| `_kd` | knowledge distillation added |
| `_enes` | English → Spanish (default is cs → de) |

## Creating a new experiment

Three ways, roughly in order of preference:

```bash
# 1. Pure CLI (no YAML file needed):
attention-lp run --exp-id foo --pruning ifr --n-remove 8 --finetune --qlora

# 2. Copy a recipe, then tweak fields:
cp recipes/ifr_prune_ft.yaml my_run.yaml
attention-lp run --config my_run.yaml

# 3. Use a recipe + flag overrides for one-off variants:
attention-lp run --config recipes/ifr_prune_ft.yaml --lang-pair en-es --n-remove 12
```
