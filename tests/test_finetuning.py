"""Tests for src/finetuning/train.py.

Tests only the pure-logic functions (create_lora_config, load_parallel_data)
without actually loading models or running training.
"""

import pytest
from pathlib import Path

from peft import LoraConfig

from src.config import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    SRC_LANG_NAME,
    TGT_LANG_NAME,
)
from src.finetuning.train import create_lora_config, load_parallel_data


# ---------------------------------------------------------------------------
# create_lora_config
# ---------------------------------------------------------------------------

class TestCreateLoraConfig:
    def test_returns_lora_config(self):
        config = create_lora_config()
        assert isinstance(config, LoraConfig)

    def test_default_values(self):
        config = create_lora_config()
        assert config.r == LORA_R
        assert config.lora_alpha == LORA_ALPHA
        assert config.lora_dropout == LORA_DROPOUT
        assert config.task_type == "CAUSAL_LM"
        assert config.bias == "none"

    def test_default_target_modules(self):
        config = create_lora_config()
        expected = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        assert set(config.target_modules) == set(expected)

    def test_custom_target_modules(self):
        custom = ["q_proj", "v_proj"]
        config = create_lora_config(target_modules=custom)
        assert set(config.target_modules) == set(custom)


# ---------------------------------------------------------------------------
# load_parallel_data
# ---------------------------------------------------------------------------

class TestLoadParallelData:
    def test_returns_dataset(self, sample_parallel_files):
        src_path, tgt_path = sample_parallel_files
        dataset = load_parallel_data(src_path, tgt_path)
        assert hasattr(dataset, "__len__")
        assert len(dataset) == 5

    def test_has_text_column(self, sample_parallel_files):
        src_path, tgt_path = sample_parallel_files
        dataset = load_parallel_data(src_path, tgt_path)
        assert "text" in dataset.column_names

    def test_text_contains_prompt(self, sample_parallel_files):
        src_path, tgt_path = sample_parallel_files
        dataset = load_parallel_data(src_path, tgt_path)
        text = dataset[0]["text"]
        assert SRC_LANG_NAME in text
        assert TGT_LANG_NAME in text

    def test_text_contains_source_and_target(self, sample_parallel_files):
        src_path, tgt_path = sample_parallel_files
        dataset = load_parallel_data(src_path, tgt_path)
        text = dataset[0]["text"]
        # Source sentence should be in the prompt
        assert "Ahoj svete" in text
        # Target should be appended
        assert "Hallo Welt" in text

    def test_all_entries_nonempty(self, sample_parallel_files):
        src_path, tgt_path = sample_parallel_files
        dataset = load_parallel_data(src_path, tgt_path)
        for item in dataset:
            assert len(item["text"]) > 0
