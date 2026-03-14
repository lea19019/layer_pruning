"""Smoke test: validates the environment, model loading, and IFR hooks.

Run this on a GPU node before submitting real experiments:
    srun --partition=cs --account=sdrich --gres=gpu:a100:1 --mem=64G --time=00:30:00 \
        bash -c "source .venv/bin/activate && python scripts/smoke_test.py"
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check_imports():
    print("1. Checking imports ...")
    import torch
    import transformers
    import peft
    import bitsandbytes
    import sacrebleu
    import sentence_transformers
    from comet import download_model
    print(f"   torch={torch.__version__}, transformers={transformers.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("   OK")


def check_model_loading():
    print("\n2. Loading Aya Expanse 8B (this takes a few minutes) ...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.utils import load_env

    load_env()

    tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-expanse-8b")
    model = AutoModelForCausalLM.from_pretrained(
        "CohereForAI/aya-expanse-8b",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"   Loaded: {model.config.num_hidden_layers} layers")
    print(f"   Architecture: {model.config.model_type}")
    print(f"   Layer type: {type(model.model.layers[0]).__name__}")
    print(f"   Has self_attn: {hasattr(model.model.layers[0], 'self_attn')}")
    print(f"   Has mlp: {hasattr(model.model.layers[0], 'mlp')}")

    # Quick generation test
    inputs = tokenizer("Translate Czech to German.\nCzech: Dobrý den\nGerman:", return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"   Generation test: {result[:100]}")
    print("   OK")

    return model, tokenizer


def check_ifr_hooks(model, tokenizer):
    print("\n3. Testing IFR hooks ...")
    from src.attribution.ifr import IFRScorer

    # Create scorer with already-loaded model
    scorer = IFRScorer.__new__(IFRScorer)
    scorer.model = model
    scorer.tokenizer = tokenizer
    scorer.device = str(model.device)
    scorer.num_layers = model.config.num_hidden_layers
    scorer._activations = {}
    scorer._hooks = []

    scores = scorer.score_single("Translate Czech to German.\nCzech: Dobrý den\nGerman: Guten Tag")

    print(f"   Layer importance shape: {scores['layer_importance'].shape}")
    print(f"   Attn importance range: [{scores['attn_importance'].min():.4f}, {scores['attn_importance'].max():.4f}]")
    print(f"   MLP importance range: [{scores['mlp_importance'].min():.4f}, {scores['mlp_importance'].max():.4f}]")
    print(f"   Layer importance range: [{scores['layer_importance'].min():.4f}, {scores['layer_importance'].max():.4f}]")

    # Sanity check: scores should be non-negative and not all zeros
    assert scores['layer_importance'].sum() > 0, "All scores are zero!"
    assert (scores['layer_importance'] >= 0).all(), "Negative scores found!"
    print("   OK")


def check_pruning(model, tokenizer):
    print("\n4. Testing layer removal ...")
    import torch
    from src.pruning.remove_layers import remove_layers

    original_layers = model.config.num_hidden_layers
    # Remove 2 middle layers as a test
    remove_layers(model, [15, 16])
    assert model.config.num_hidden_layers == original_layers - 2

    # Verify model still generates
    inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10)
    print(f"   Pruned model generates: {tokenizer.decode(output[0], skip_special_tokens=True)[:50]}")
    print("   OK")


def main():
    print("=" * 50)
    print("SMOKE TEST")
    print("=" * 50)

    check_imports()
    model, tokenizer = check_model_loading()
    check_ifr_hooks(model, tokenizer)
    check_pruning(model, tokenizer)

    print("\n" + "=" * 50)
    print("ALL CHECKS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
