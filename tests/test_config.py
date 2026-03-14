"""Tests for src/config.py -- verify config constants have correct types/values."""

from pathlib import Path

from src.config import (
    BASE_MODEL,
    COMET_MODEL,
    CONFIGS_DIR,
    DATA_DIR,
    EXPERIMENTS_DIR,
    FILTERED_DIR,
    FT_BATCH_SIZE,
    FT_EPOCHS,
    FT_GRAD_ACCUM,
    FT_LEARNING_RATE,
    KD_COMET_THRESHOLD,
    KD_DIR,
    LANG_DETECT_THRESHOLD,
    LENGTH_RATIO_MAX,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_WORDS_PER_SEGMENT,
    NUM_LAYERS,
    PROJECT_ROOT,
    PRUNE_TARGETS,
    RAW_DIR,
    RESULTS_DIR,
    SEMANTIC_MODEL,
    SEMANTIC_SIM_THRESHOLD,
    SRC_LANG,
    SRC_LANG_NAME,
    TGT_LANG,
    TGT_LANG_NAME,
    TEST_SIZE,
    TRAIN_SIZE,
    TRANSLATION_PROMPT,
)


class TestPaths:
    def test_project_root_is_path(self):
        assert isinstance(PROJECT_ROOT, Path)

    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()

    def test_data_dir_under_root(self):
        assert DATA_DIR == PROJECT_ROOT / "data"

    def test_raw_dir_under_data(self):
        assert RAW_DIR == DATA_DIR / "raw"

    def test_filtered_dir_under_data(self):
        assert FILTERED_DIR == DATA_DIR / "filtered"

    def test_kd_dir_under_data(self):
        assert KD_DIR == DATA_DIR / "kd"

    def test_experiments_dir_under_root(self):
        assert EXPERIMENTS_DIR == PROJECT_ROOT / "experiments"

    def test_configs_dir(self):
        assert CONFIGS_DIR == EXPERIMENTS_DIR / "configs"

    def test_results_dir(self):
        assert RESULTS_DIR == EXPERIMENTS_DIR / "results"


class TestModelConstants:
    def test_base_model_is_string(self):
        assert isinstance(BASE_MODEL, str)

    def test_base_model_value(self):
        assert BASE_MODEL == "CohereForAI/aya-expanse-8b"

    def test_num_layers_type(self):
        assert isinstance(NUM_LAYERS, int)

    def test_num_layers_value(self):
        assert NUM_LAYERS == 32

    def test_comet_model_is_string(self):
        assert isinstance(COMET_MODEL, str)


class TestLanguagePair:
    def test_src_lang(self):
        assert SRC_LANG == "ces"

    def test_tgt_lang(self):
        assert TGT_LANG == "deu"

    def test_src_lang_name(self):
        assert SRC_LANG_NAME == "Czech"

    def test_tgt_lang_name(self):
        assert TGT_LANG_NAME == "German"


class TestFilteringThresholds:
    def test_max_words_per_segment(self):
        assert isinstance(MAX_WORDS_PER_SEGMENT, int)
        assert MAX_WORDS_PER_SEGMENT == 200

    def test_length_ratio_max(self):
        assert isinstance(LENGTH_RATIO_MAX, (int, float))
        assert LENGTH_RATIO_MAX == 1.5

    def test_lang_detect_threshold(self):
        assert 0 < LANG_DETECT_THRESHOLD <= 1.0

    def test_semantic_sim_threshold(self):
        assert 0 < SEMANTIC_SIM_THRESHOLD <= 1.0

    def test_semantic_model_is_string(self):
        assert isinstance(SEMANTIC_MODEL, str)


class TestDatasetSplits:
    def test_train_size(self):
        assert isinstance(TRAIN_SIZE, int)
        assert TRAIN_SIZE == 100_000

    def test_test_size(self):
        assert isinstance(TEST_SIZE, int)
        assert TEST_SIZE == 500


class TestKDConfig:
    def test_kd_comet_threshold(self):
        assert isinstance(KD_COMET_THRESHOLD, float)
        assert 0 < KD_COMET_THRESHOLD <= 1.0


class TestPruningTargets:
    def test_prune_targets_is_list(self):
        assert isinstance(PRUNE_TARGETS, list)

    def test_prune_targets_values(self):
        assert PRUNE_TARGETS == [8, 12, 16]

    def test_prune_targets_all_int(self):
        assert all(isinstance(t, int) for t in PRUNE_TARGETS)


class TestFineTuningDefaults:
    def test_learning_rate(self):
        assert isinstance(FT_LEARNING_RATE, float)
        assert FT_LEARNING_RATE > 0

    def test_epochs(self):
        assert isinstance(FT_EPOCHS, int)
        assert FT_EPOCHS > 0

    def test_batch_size(self):
        assert isinstance(FT_BATCH_SIZE, int)
        assert FT_BATCH_SIZE > 0

    def test_grad_accum(self):
        assert isinstance(FT_GRAD_ACCUM, int)
        assert FT_GRAD_ACCUM > 0

    def test_lora_r(self):
        assert isinstance(LORA_R, int)
        assert LORA_R > 0

    def test_lora_alpha(self):
        assert isinstance(LORA_ALPHA, int)
        assert LORA_ALPHA > 0

    def test_lora_dropout(self):
        assert isinstance(LORA_DROPOUT, float)
        assert 0 <= LORA_DROPOUT < 1.0


class TestTranslationPrompt:
    def test_prompt_is_string(self):
        assert isinstance(TRANSLATION_PROMPT, str)

    def test_prompt_has_placeholders(self):
        assert "{src_lang}" in TRANSLATION_PROMPT
        assert "{tgt_lang}" in TRANSLATION_PROMPT
        assert "{source}" in TRANSLATION_PROMPT

    def test_prompt_can_be_formatted(self):
        result = TRANSLATION_PROMPT.format(
            src_lang="Czech", tgt_lang="German", source="Ahoj"
        )
        assert "Czech" in result
        assert "German" in result
        assert "Ahoj" in result
