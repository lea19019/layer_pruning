# WMT 2025 Model Compression Shared Task — Key Findings

Source: [Gaido et al., 2025](https://aclanthology.org/2025.wmt-1.25/)

## Submissions

| Team | Submission | Method | Quant | Inference | cs-de COMET |
|------|-----------|--------|-------|-----------|-------------|
| Stevens/Rice/Lambda | **AyaQ** | GPTQ 4-bit (group_size=32) | GPTQ | vLLM | 40.3 |
| Stevens/Rice/Lambda | **LeanAya** | LeanQuant (learned grids) | Custom | unknown | 53.2 |
| TCD (Moslem) | **TCD-Kreasof** primary | Iterative pruning → 24L + FT | None (fp16) | HF/vLLM | 39.9 |
| TCD (Moslem) | contrastive1 | Iterative pruning → 20L + FT | None | HF/vLLM | 32.4 |
| TCD (Moslem) | contrastive2 | Iterative pruning → 16L + FT | None | HF/vLLM | 21.4 |
| Vicomtech | gelaco-0.25_ft_q4 | Layer merging (25%) + FT + BnB 4bit | BnB | BnB | 41.2 |
| Vicomtech | gelaco-0.25_ft_q8 | Layer merging (25%) + FT + BnB 8bit | BnB | BnB | 44.4 |
| Vicomtech | gelaco-0.50_ft_q4 | Layer merging (50%) + FT + BnB 4bit | BnB | BnB | 31.0 |
| Vicomtech | gelaco-0.75_gkd_q4 | Layer merging (75%) + GKD + BnB 4bit | BnB | BnB | 31.1 |
| Organizers | base | Aya 8B fp16 (unmodified) | None | HF | 55.3 |
| Organizers | bnb-8bit | BnB 8-bit | BnB | HF | 55.6 |
| Organizers | bnb-4bit-nf4 | BnB 4-bit NF4 | BnB | HF | 55.7 |
| Organizers | bnb-4bit-nf4-2q | BnB 4-bit NF4 double-quant | BnB | HF | 55.5 |

## Speed Results (Figure 1, batch_size=16, cs-de)

| System | ~tok/s | Notes |
|--------|--------|-------|
| AyaQ (GPTQ 4-bit + vLLM) | ~400 | **Fastest by far** |
| LeanAya | ~350 | Learned quantization grids |
| Aya8B base (fp16) | ~150 | Baseline |
| TCD-Kreasof primary (24L, fp16) | ~200 | Pruning speedup |
| Vicomtech gelaco-0.75_gkd_q4 | ~100 | BnB quantization is slow |
| BitsAndBytes bnb-8bit | ~80 | Slower than fp16 baseline |

## Key Findings

1. **AyaQ (GPTQ + vLLM) is 2.7x faster than base** and fastest overall
2. **BnB quantization is SLOWER than fp16** — confirmed by organizer baselines
3. **Pruning (Moslem) gives moderate speedup** (~1.3x) but loses quality
4. **GPTQ quality is lower** — AyaQ COMET 40.3 vs base 55.3 (significant drop at 4-bit)
5. **LeanAya best quality-speed tradeoff** — COMET 53.2 at ~350 tok/s
6. Speed measured on A100 80GB, batch sizes 1-512, 3 runs averaged

## Hardware

- Single Nvidia A100 80GB VRAM
- AMD EPYC CPU, 96 cores, 866GB RAM
- Docker containers with all software included

## Evaluation

- Speed: average output tokens/second across full test set, 3 runs
- Quality: XCOMET-XL and MetricX-24-Hybrid-XL
- Size: disk footprint in GB

---

# Quantization Technologies Explained

## BitsAndBytes (BnB)

**What:** A library by Tim Dettmers that enables loading models in INT8 or INT4 precision at runtime. Widely used via HuggingFace's `load_in_8bit` / `load_in_4bit` flags.

**How it works:** Stores weights in low-bit format to save GPU memory. During each forward pass, it **dequantizes weights to fp16** on-the-fly, performs the matrix multiplication, then discards the fp16 copy. This means every layer's weights get decompressed every single time they're used.

**Speed:** **Slower than fp16** on high-VRAM GPUs because of the dequantization overhead. The round-trip (load INT8 → dequantize → compute → discard) adds latency to every layer. Beneficial only when the model wouldn't fit in GPU memory otherwise.

**Quality:** Minimal loss at 8-bit. Some loss at 4-bit (NF4/FP4). No calibration needed — just load and run.

**Our usage:** All our INT8 experiments use BnB. Good for quality measurement, bad for speed measurement.

## GPTQ (GPT Quantization)

**What:** A post-training quantization method (Frantar et al., 2023) that compresses model weights to 4-bit or 8-bit using calibration data. The key difference from BnB: GPTQ produces a **pre-quantized model** with optimized weight representations.

**How it works:** Uses a second-order optimization (based on the inverse Hessian of the weight matrices) to find the best low-bit representation of each weight group. Requires a small calibration dataset (32-128 examples) to measure the impact of quantization on each layer. The process runs once offline; the result is a static quantized model.

**Speed:** **Faster than fp16** when served with vLLM or ExLlama. These engines have custom CUDA kernels (Marlin, Machete) that compute directly on 4-bit weights without dequantization. This is why AyaQ (GPTQ + vLLM) is 2.7x faster than fp16 in the WMT shared task.

**Quality:** Depends on bit width and group size. 4-bit with group_size=128 typically loses 1-3% quality. 8-bit is nearly lossless. The calibration data matters — using task-relevant data (translation examples) gives better results.

**Key parameters:**
- `bits`: 4 or 8
- `group_size`: 32, 64, 128 (smaller = better quality, larger model, slower)
- `desc_act`: whether to use activation order (slower export, slightly better quality)

## AWQ (Activation-Aware Weight Quantization)

**What:** Similar to GPTQ but focuses on protecting the most important weights. Won MLSys 2024 Best Paper Award.

**How it works:** Observes that a small fraction of weights (salient weights) disproportionately affect model quality. Instead of quantizing all weights equally, AWQ identifies salient weights based on activation magnitudes and applies per-channel scaling to protect them before quantization. This means important weights get more precision without increasing the overall bit budget.

**Speed:** Comparable to GPTQ when served with vLLM (uses same Marlin kernels). ~3x faster than fp16 at 4-bit.

**Quality:** Generally slightly better than GPTQ at the same bit width because of the activation-aware protection of salient weights.

## LLM Compressor (vLLM Project)

**What:** A library from the vLLM team for exporting quantized models in formats compatible with vLLM's optimized inference kernels. Used by AyaQ in the WMT shared task.

**How it works:** Takes a fp16 model, applies GPTQ/AWQ/SparseGPT quantization with calibration data, and outputs a model in a format that vLLM can load and serve with maximum throughput.

**URL:** https://github.com/vllm-project/llm-compressor

## LeanQuant

**What:** A quantization method that learns optimal quantization grids rather than using fixed affine mappings. Used by LeanAya in the WMT shared task.

**How it works:** Standard quantization maps continuous weights to discrete levels using fixed, evenly-spaced grids. LeanQuant instead learns the grid positions by minimizing quantization error through backpropagation. This is especially helpful when weight distributions have outliers that would distort a fixed grid.

**Speed:** ~350 tok/s in the shared task, slightly slower than GPTQ+vLLM.

**Quality:** Better than GPTQ at same bit width — LeanAya COMET 53.2 vs AyaQ 40.3.

## vLLM

**What:** A high-throughput LLM inference engine that uses PagedAttention for efficient memory management. Supports GPTQ, AWQ, and other quantization formats with optimized CUDA kernels.

**How it works:** Instead of allocating a contiguous block of GPU memory for each request's KV cache, vLLM uses a paged allocation strategy (like virtual memory in operating systems). This dramatically reduces memory waste from fragmentation, enabling higher batch sizes and throughput.

**Speed advantage:** For quantized models, vLLM's Marlin kernels perform matrix multiplication directly on 4-bit weights without dequantization. Combined with PagedAttention for efficient batching, this gives 2-4x speedup over HuggingFace Transformers.

**Our usage:** We already use vLLM for KD data generation (Aya 32B). For speed benchmarking, we now use it to serve both fp16 and GPTQ-quantized models.

## Marlin Kernels

**What:** Custom CUDA kernels for fast INT4 matrix multiplication, integrated into vLLM. Named after "Mixed-precision Accelerated Routines for LLM INference."

**How it works:** Standard GPU matrix multiplication expects fp16/fp32 inputs. Marlin kernels operate directly on packed INT4 data, performing the multiply-accumulate operations without unpacking to fp16 first. This eliminates the dequantization bottleneck that makes BnB slow.

**Speed:** Up to 4x faster than fp16 for INT4 models. Only activated when loading GPTQ/AWQ models through vLLM.

---

# Implications for Our Project

1. **Our BnB INT8 speed numbers are meaningless for real deployment.** BnB is slower than fp16. Use GPTQ + vLLM instead.

2. **We should export all finetuned models to GPTQ 4-bit and 8-bit**, then benchmark with vLLM. This gives us speed numbers comparable to the WMT shared task.

3. **We have the fp16 checkpoints needed** for GPTQ export (I1, I2, M1, M2, B2, B3 finetuned/merged models). The BnB quantized checkpoints (I3, I4, M3, M4) can't be converted, but the underlying fp16 models are the same as I1/I2/M1/M2.

4. **GPTQ may reduce quality** — AyaQ dropped from 55.3 to 40.3 COMET at 4-bit. We need to measure quality alongside speed to find the right trade-off.

5. **Pruning + GPTQ could be the best combination** — fewer layers (faster) + quantized kernels (faster) + smaller size. Nobody in the shared task tried this. We should.
