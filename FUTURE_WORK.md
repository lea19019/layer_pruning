# Future Work: Interpretability-Guided Mixed-Precision Quantization

## Core Idea

Instead of removing layers entirely (which causes quality cliffs at high compression), use interpretability signals to assign **different quantization precision per layer**: important layers get high precision (8-bit), less important layers get aggressive compression (3-4 bit). All 32 layers are preserved, avoiding the catastrophic quality drops we see at 16 layers removed.

This extends our current work from "interpretability for pruning decisions" to "interpretability for precision allocation decisions" — a more general framework.

## Proposed Approach

### Phase 1: Multiple Attribution Methods

Score each layer's importance using several interpretability methods, then compare which produces the best signal for precision allocation.

| Method | Signal | Cost | Implementation |
|--------|--------|------|---------------|
| **IFR** (already done) | L1 proximity on residual stream | 1 forward pass | `src/attribution/ifr.py` |
| **Activation magnitude** | ‖layer_output‖ averaged across examples | 1 forward pass | ~20 lines, reuse hook infrastructure |
| **Gradient norm** | ‖∂L/∂layer_output‖ | 1 forward + 1 backward | ~30 lines, standard autograd |
| **Attention entropy** | H(softmax(QK^T)) per layer | 1 forward pass | ~30 lines, hook on attention weights |
| **LRP** | Backward relevance propagation | 1 forward + 1 backward | ~200+ lines, custom rules for Cohere arch |

Priority: IFR, activation magnitude, gradient norm, attention entropy first. LRP if time permits.

### Phase 2: Mixed-Precision Quantization

Given layer importance scores, assign bit widths:
- **Tier 1** (most important ~10 layers): 8-bit
- **Tier 2** (middle ~12 layers): 4-bit
- **Tier 3** (least important ~10 layers): 3-bit or 2-bit

Implementation options:
- **GPTQ via AutoGPTQ** — supports per-layer bit config, well-tested
- **AWQ** — activation-aware, MLSys 2024 Best Paper
- **GGUF export** — supports mixed precision natively, CPU/GPU inference

### Phase 3: Evaluation

Compare against:
1. Uniform 8-bit (current B1_int8)
2. Uniform 4-bit (current B1 INT4)
3. Layer pruning + LoRA (current I1 experiments)
4. Layer pruning + LoRA + uniform INT8 (current I3_int8 experiments)

Metrics: COMET, chrF++, BLEU, model size (MB), inference speed.
Language pairs: Czech→German, English→Spanish.

### Phase 4: Cross-Method Analysis

The key research question: **does the "best" attribution method for mixed-precision quantization match the "best" for layer pruning?** IFR identifies layers 8-18 as least important for pruning, but those same layers might tolerate aggressive quantization differently than layers that are important for pruning.

## Related Work

### Mixed-Precision Quantization for LLMs

- **[Mixed-Precision Quantization for Language Models: Techniques and Prospects](https://arxiv.org/abs/2510.16805)** (2025) — Comprehensive survey of mixed-precision methods. Key insight: "layer-wise precision assignment is computationally efficient but inadequate for LLMs where sensitive weights are sparsely distributed." Our interpretability-based approach could address this by identifying sensitivity at a more meaningful level than simple statistical measures.

- **[Variable Layer-Wise Quantization](https://arxiv.org/abs/2406.17415)** (2024) — Proposes two layer importance strategies: (1) how different output embeddings are from input embeddings, (2) number of outlier weights per layer. Shows that with proper ordering, 90% performance is retained at 2.85 average bits. **Directly relevant** — but they use simple statistical measures for importance, not interpretability signals.

- **[RAMP: Reinforcement Adaptive Mixed Precision](https://arxiv.org/abs/2603.17891)** (2026) — Uses reinforcement learning (Soft Actor-Critic) to learn per-layer bit widths. Achieves 5.54 perplexity at 3.65 effective bits on Llama-2-7B, outperforming uniform 4-bit. Key finding: "quantization sensitivity is primarily architectural" — policies transfer zero-shot across model families. **Our approach differs** by using interpretability signals (task-specific, translation-focused) rather than RL (task-agnostic, perplexity-based).

- **[CALM: CKA-Guided Adaptive Layer-Wise Modularization](https://arxiv.org/html/2512.16282)** (2025) — Uses Centered Kernel Alignment to measure layer similarity and guide per-layer algorithm selection (GPTQ vs AWQ). Does not learn bit widths.

- **[MixLLM](https://arxiv.org/abs/2412.14590)** (2024) — Global mixed-precision between output features with efficient CUDA kernels. Focuses on output-feature granularity rather than layer-level.

- **[Saliency Assisted Quantization](https://arxiv.org/html/2411.05858v1)** (2024) — Uses saliency maps to guide quantization decisions. **Closest to our idea** — but applied to CNNs/general models, not LLM translation. Combining saliency with IFR for translation-specific quantization would be novel.

- **[Precision Where It Matters: Spike-Aware Mixed-Precision](https://arxiv.org/html/2504.21553v1)** (2025) — Observes activation spikes concentrated in specific LLaMA layers, applies FP16 to those and low-bit to rest. Task-agnostic approach based on activation statistics.

- **[You Had One Job: Per-Task Quantization](https://arxiv.org/html/2511.06516)** (2025) — Uses hidden representations to determine per-layer precision allocation that is **task-specific**. Very relevant — they show different tasks benefit from different precision allocations. Our work would extend this to translation with interpretability-based importance.

- **[MoQAE: Mixed-Precision for Long-Context](https://aclanthology.org/2025.acl-long.531.pdf)** (2025, ACL) — Mixed-precision KV-cache compression for long-context inference. Different application but similar principle of non-uniform precision.

- **[Exploring Layer-wise Information Effectiveness for PTQ](https://arxiv.org/pdf/2508.03332)** (2025) — Studies how information flows through layers during quantization in small language models. Directly combines information flow analysis with quantization decisions.

### Attribution-Guided Pruning (Related)

- **[Vakilzadeh Hatefi et al.](https://arxiv.org/abs/2506.13727)** (2025) — LRP for unstructured pruning of LLMs. Parameter-level attribution. We would use similar attribution methods but for precision allocation rather than pruning.

- **[Ferrando & Voita, IFR](https://aclanthology.org/2024.emnlp-main.305/)** (EMNLP 2024) — Information Flow Routes. Our current attribution method. Could be extended from pruning decisions to quantization decisions.

- **[Moslem et al.](https://aclanthology.org/2025.wmt-1.78/)** (WMT 2025) — Iterative layer pruning for translation. Our baseline comparison. Mixed-precision quantization is an alternative compression strategy that preserves all layers.

## What Makes This Novel

1. **Interpretability → quantization precision** — existing work uses statistical measures (outlier counts, activation distributions) or RL. Using IFR/gradient/attention-entropy for precision allocation is new.

2. **Translation-specific** — most mixed-precision work evaluates on perplexity or general benchmarks. We'd evaluate on translation quality (COMET), which is a more demanding and task-specific metric.

3. **Multiple attribution methods compared** — systematic evaluation of which interpretability signal best predicts optimal per-layer precision for translation.

4. **Cross-lingual** — testing on Czech→German and English→Spanish to see if optimal precision allocations are language-pair dependent or universal.

5. **Combined with pruning** — mixed-precision quantization on already-pruned models (e.g., IFR prune 8 layers, then mixed-precision the remaining 24) could achieve extreme compression.

## Implementation Estimate

| Component | Effort | Dependencies |
|-----------|--------|-------------|
| Activation magnitude scorer | 1 day | Existing hook infrastructure |
| Gradient norm scorer | 1 day | Existing hook infrastructure |
| Attention entropy scorer | 1 day | Existing hook infrastructure |
| LRP scorer | 5-7 days | Custom Cohere propagation rules |
| GPTQ mixed-precision module | 2-3 days | AutoGPTQ library |
| Experiment configs + SLURM | 1 day | Existing framework |
| Running experiments | 3-5 days | Cluster availability |
| Analysis + plots | 1-2 days | Existing plot infrastructure |

**Total: ~2-3 weeks** (without LRP), **~4 weeks** (with LRP)
