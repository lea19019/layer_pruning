"""Translation generation using HuggingFace or vLLM."""

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


def translate_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
    batch_size: int = 8,
    temperature: float = 0.0,
) -> list[str]:
    """Generate translations for a batch of prompts using HuggingFace generate.

    Args:
        model: Loaded CausalLM model.
        tokenizer: Model tokenizer.
        prompts: List of translation prompts.
        max_new_tokens: Maximum tokens to generate.
        batch_size: Batch size for generation.
        temperature: Sampling temperature (0.0 = greedy).

    Returns:
        List of generated translations.
    """
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left-pad for correct generation

    all_translations = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Translating"):
        batch = prompts[i : i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # Decode only the generated part for each sequence individually.
        # Each sequence in the batch may have a different prompt length
        # due to left-padding, so we track per-sequence input length.
        for j, output in enumerate(outputs):
            # The input length for this specific sequence (including padding)
            input_len = inputs["input_ids"].shape[1]
            generated = output[input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            all_translations.append(text)

    return all_translations


def translate_with_vllm(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int = 256,
    tensor_parallel_size: int = 1,
    dtype: str = "float16",
) -> list[str]:
    """Generate translations using vLLM for faster inference.

    Args:
        model_path: Path to the model.
        prompts: List of translation prompts.
        max_new_tokens: Maximum tokens to generate.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        dtype: Model dtype.

    Returns:
        List of generated translations.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        trust_remote_code=True,
    )
    params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
    )

    outputs = llm.generate(prompts, params)
    translations = [out.outputs[0].text.strip() for out in outputs]
    return translations
