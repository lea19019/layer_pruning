"""Fine-tuning script using LoRA/QLoRA with PEFT and TRL."""

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from src.config import (
    BASE_MODEL,
    FT_BATCH_SIZE,
    FT_EPOCHS,
    FT_GRAD_ACCUM,
    FT_LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    SRC_LANG_NAME,
    TGT_LANG_NAME,
    TRANSLATION_PROMPT,
)
from src.utils import load_env, set_seed


def load_parallel_data(src_path: Path, tgt_path: Path) -> Dataset:
    """Load parallel text files into a HuggingFace Dataset."""
    with open(src_path) as f:
        sources = f.read().splitlines()
    with open(tgt_path) as f:
        targets = f.read().splitlines()

    texts = []
    for src, tgt in zip(sources, targets):
        prompt = TRANSLATION_PROMPT.format(
            src_lang=SRC_LANG_NAME,
            tgt_lang=TGT_LANG_NAME,
            source=src,
        )
        texts.append(prompt + " " + tgt)

    return Dataset.from_dict({"text": texts})


def create_lora_config(
    target_modules: list[str] | None = None,
) -> LoraConfig:
    """Create LoRA configuration for Cohere/Aya models."""
    if target_modules is None:
        # Standard targets for Cohere architecture
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def finetune(
    model_name_or_path: str,
    train_src: Path,
    train_tgt: Path,
    output_dir: Path,
    use_qlora: bool = False,
    epochs: int = FT_EPOCHS,
    batch_size: int = FT_BATCH_SIZE,
    grad_accum: int = FT_GRAD_ACCUM,
    learning_rate: float = FT_LEARNING_RATE,
    max_seq_length: int = 512,
):
    """Fine-tune a (possibly pruned) model with LoRA.

    Args:
        model_name_or_path: HuggingFace model or path to pruned model.
        train_src: Path to training source sentences.
        train_tgt: Path to training target sentences.
        output_dir: Where to save the fine-tuned model.
        use_qlora: If True, load model in 4-bit and use QLoRA.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        grad_accum: Gradient accumulation steps.
        learning_rate: Learning rate.
        max_seq_length: Maximum sequence length.
    """
    load_env()
    set_seed(42)

    # Load dataset
    dataset = load_parallel_data(train_src, train_tgt)
    print(f"Training on {len(dataset)} examples")

    # Model loading
    model_kwargs = {
        "dtype": torch.float16,
        "device_map": "auto",
    }

    if use_qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    # LoRA
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Training args
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    trainer.train()

    # Save merged model
    merged_dir = output_dir / "merged"
    print(f"Saving merged model to {merged_dir}")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    return merged_dir


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model with LoRA")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--train-src", type=Path, required=True)
    parser.add_argument("--train-tgt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--epochs", type=int, default=FT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=FT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=FT_LEARNING_RATE)
    args = parser.parse_args()

    finetune(
        model_name_or_path=args.model,
        train_src=args.train_src,
        train_tgt=args.train_tgt,
        output_dir=args.output_dir,
        use_qlora=args.qlora,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
