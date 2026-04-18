#!/bin/bash
#SBATCH --job-name=fix_enes
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/fix_enes_%j.out
#SBATCH --error=logs/fix_enes_%j.err

# Fix failed en-es KD experiments:
# - I2_12_enes: model saved, just needs eval
# - I2_16_enes: full rerun (failed during training setup)
# - M2_8_enes: full rerun (failed during training setup)
# Then benchmark all 5 en-es KD experiments (including M2_12/16 from other job)

set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "=== Fix en-es KD experiments ==="
echo "=== Started at $(date) ==="

# 1. Eval-only for I2_12_enes (model already trained and saved)
echo ""
echo "============================================================"
echo "=== Eval-only: I2_12_enes at $(date) ==="
echo "============================================================"
python -c "
import sys
sys.path.insert(0, '.')
from src.evaluation.run_eval import evaluate_model
from src.run_experiment import _resolve_lang_pair, _data_dir, LANG_DEFAULTS
from src.config import TRANSLATION_PROMPT
import yaml, json
from pathlib import Path

exp_id = 'I2_12_enes'
config_path = 'experiments/configs/{}.yaml'.format(exp_id)
with open(config_path) as f:
    cfg = yaml.safe_load(f)

lang = _resolve_lang_pair(cfg)
data_dir = _data_dir(lang)
model_path = 'experiments/results/{}/finetuned/merged'.format(exp_id)
results_dir = Path('experiments/results/{}'.format(exp_id))
results_dir.mkdir(parents=True, exist_ok=True)

from src.evaluation.translate import translate_batch
from src.evaluation.metrics import compute_comet, compute_chrf, compute_bleu
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print('Loading model from', model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')

with open(data_dir / 'test.{}'.format(lang['src_code'])) as f:
    sources = f.read().splitlines()
with open(data_dir / 'test.{}'.format(lang['tgt_code'])) as f:
    references = f.read().splitlines()

prompts = [TRANSLATION_PROMPT.format(src_lang=lang['src_name'], tgt_lang=lang['tgt_name'], source=s) for s in sources]
hypotheses = translate_batch(model, tokenizer, prompts, batch_size=20, max_new_tokens=256)

comet = compute_comet(hypotheses, references, sources)
chrf = compute_chrf(hypotheses, references)
bleu = compute_bleu(hypotheses, references)

print('COMET={:.4f} chrF++={:.2f} BLEU={:.2f}'.format(comet, chrf, bleu))

import os
model_size = sum(f.stat().st_size for f in Path(model_path).rglob('*') if f.is_file()) / (1024**3)

results = {
    'config': cfg,
    'metrics': {
        'comet': comet,
        'chrf': chrf,
        'bleu': bleu,
        'model_size_gb': round(model_size, 2),
    }
}
with open(results_dir / 'results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print('Results saved.')
" || echo "=== FAILED I2_12_enes eval ==="

# 2. Full rerun for I2_16_enes and M2_8_enes
for exp in I2_16_enes M2_8_enes; do
    RESULTS="experiments/results/${exp}/results.json"
    if [ -f "$RESULTS" ]; then
        echo "=== SKIP ${exp}: results already exist ==="
        continue
    fi
    echo ""
    echo "============================================================"
    echo "=== Running ${exp} at $(date) ==="
    echo "============================================================"
    python -m src.run_experiment "experiments/configs/${exp}.yaml" || {
        echo "=== FAILED ${exp} (exit $?) — continuing ==="
    }
done

# 3. Check if M2_12_enes and M2_16_enes finished from the other job, run if not
for exp in M2_12_enes M2_16_enes; do
    RESULTS="experiments/results/${exp}/results.json"
    if [ -f "$RESULTS" ]; then
        echo "=== SKIP ${exp}: results already exist ==="
        continue
    fi
    echo ""
    echo "============================================================"
    echo "=== Running ${exp} at $(date) ==="
    echo "============================================================"
    python -m src.run_experiment "experiments/configs/${exp}.yaml" || {
        echo "=== FAILED ${exp} (exit $?) — continuing ==="
    }
done

# 4. Run GPTQ + vLLM benchmark for all new experiments
echo ""
echo "============================================================"
echo "=== Running speed benchmarks at $(date) ==="
echo "============================================================"
python scripts/benchmark_speed.py

echo ""
echo "=== All done at $(date) ==="
