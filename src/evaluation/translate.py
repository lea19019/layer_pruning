"""Translation generation using HuggingFace or vLLM."""

import re

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.config import SRC_LANG_NAME

# Stop strings that signal the model has finished translating and is
# starting to hallucinate additional content (e.g. new source/target pairs,
# meta-commentary). Used both as generation stop strings and as a
# post-processing safety net.
STOP_STRINGS = [f"\n{SRC_LANG_NAME}:", "\nCzech:", "\n\n"]
# Broader regex for post-processing: also catches "\nTranslation:",
# "\n(Translation", "\nGerman:", and other common continuation patterns.
_STOP_PATTERN = re.compile(
    r"\nCzech:|"
    r"\n" + re.escape(SRC_LANG_NAME) + r":|"
    r"\nGerman:|"
    r"\nTranslation[:\s(]|"
    r"\n\(Translation|"
    r"\n\n"
)


def _extract_translation(text: str) -> str:
    """Extract the first translation from generated text.

    The model sometimes continues generating after the translation (e.g.
    additional language pairs, meta-commentary). This truncates at the
    first stop pattern and strips whitespace.
    """
    m = _STOP_PATTERN.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


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
    # Left-padding is required for causal LMs: right-padding would place
    # pad tokens between the real prompt tokens and the generated tokens,
    # breaking autoregressive generation. Left-padding keeps all real tokens
    # contiguous at the right edge where generation begins.
    tokenizer.padding_side = "left"

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
            stop_strings=STOP_STRINGS,
            tokenizer=tokenizer,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # Strip the prompt from each output to get only the generated text.
        # Because we used left-padding, all sequences in the batch share the
        # same padded input length (shape[1]), so we can use a single offset
        # to slice off the prompt portion from every output sequence.
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"].shape[1]
            generated = output[input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            text = _extract_translation(text)
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
        stop=STOP_STRINGS,
    )

    outputs = llm.generate(prompts, params)
    translations = [
        _extract_translation(out.outputs[0].text) for out in outputs
    ]
    return translations
