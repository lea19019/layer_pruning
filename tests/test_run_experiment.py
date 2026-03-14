"""Tests for src/run_experiment.py -- config loading and validation."""

import yaml

from src.utils import load_experiment_config


class TestExperimentConfigLoading:
    def test_loads_valid_config(self, sample_yaml_config):
        cfg = load_experiment_config(sample_yaml_config)
        assert cfg["experiment_id"] == "T1"
        assert cfg["seed"] == 42

    def test_pruning_section(self, sample_yaml_config):
        cfg = load_experiment_config(sample_yaml_config)
        assert "pruning" in cfg
        assert cfg["pruning"]["method"] == "none"

    def test_finetuning_section(self, sample_yaml_config):
        cfg = load_experiment_config(sample_yaml_config)
        assert "finetuning" in cfg
        assert cfg["finetuning"]["enabled"] is False

    def test_distillation_section(self, sample_yaml_config):
        cfg = load_experiment_config(sample_yaml_config)
        assert "distillation" in cfg
        assert cfg["distillation"]["enabled"] is False

    def test_quantization_section(self, sample_yaml_config):
        cfg = load_experiment_config(sample_yaml_config)
        assert "quantization" in cfg
        assert cfg["quantization"]["enabled"] is False


class TestExperimentConfigVariants:
    def test_ifr_pruning_config(self, tmp_path):
        cfg = {
            "experiment_id": "I1_8",
            "description": "IFR pruning test",
            "base_model": "test-model",
            "seed": 42,
            "pruning": {
                "method": "ifr",
                "n_remove": 8,
                "scores_path": "experiments/results/ifr_scores.json",
            },
            "finetuning": {"enabled": True, "epochs": 3, "qlora": False},
            "distillation": {"enabled": False},
            "quantization": {"enabled": False},
        }
        path = tmp_path / "ifr.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)

        loaded = load_experiment_config(path)
        assert loaded["pruning"]["method"] == "ifr"
        assert loaded["pruning"]["n_remove"] == 8
        assert loaded["finetuning"]["enabled"] is True

    def test_heuristic_pruning_config(self, tmp_path):
        cfg = {
            "experiment_id": "M1_8",
            "description": "Heuristic pruning test",
            "base_model": "test-model",
            "seed": 42,
            "pruning": {
                "method": "heuristic",
                "target_layers": 24,
            },
            "finetuning": {"enabled": False},
            "distillation": {"enabled": False},
            "quantization": {"enabled": False},
        }
        path = tmp_path / "heuristic.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)

        loaded = load_experiment_config(path)
        assert loaded["pruning"]["method"] == "heuristic"
        assert loaded["pruning"]["target_layers"] == 24

    def test_kd_config(self, tmp_path):
        cfg = {
            "experiment_id": "I3_8",
            "description": "IFR + KD test",
            "base_model": "test-model",
            "seed": 42,
            "pruning": {"method": "ifr", "n_remove": 8},
            "finetuning": {"enabled": True, "epochs": 3, "qlora": False},
            "distillation": {"enabled": True},
            "quantization": {"enabled": False},
        }
        path = tmp_path / "kd.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)

        loaded = load_experiment_config(path)
        assert loaded["distillation"]["enabled"] is True
        assert loaded["finetuning"]["enabled"] is True

    def test_quantization_config(self, tmp_path):
        cfg = {
            "experiment_id": "I4_8",
            "description": "IFR + FT + Quant",
            "base_model": "test-model",
            "seed": 42,
            "pruning": {"method": "ifr", "n_remove": 8},
            "finetuning": {"enabled": True},
            "distillation": {"enabled": False},
            "quantization": {"enabled": True},
        }
        path = tmp_path / "quant.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)

        loaded = load_experiment_config(path)
        assert loaded["quantization"]["enabled"] is True


class TestRealConfigFiles:
    """Verify the actual config files in experiments/configs/ are valid YAML."""

    def test_real_configs_parse(self):
        from src.config import CONFIGS_DIR

        if not CONFIGS_DIR.exists():
            return  # skip if configs dir doesn't exist

        yaml_files = list(CONFIGS_DIR.glob("*.yaml"))
        assert len(yaml_files) > 0, "Expected at least one config file"

        for yf in yaml_files:
            cfg = load_experiment_config(yf)
            assert isinstance(cfg, dict), f"{yf.name} did not parse as dict"
            assert "experiment_id" in cfg, f"{yf.name} missing experiment_id"
            assert "pruning" in cfg, f"{yf.name} missing pruning section"

    def test_real_configs_have_required_fields(self):
        from src.config import CONFIGS_DIR

        if not CONFIGS_DIR.exists():
            return

        required = {"experiment_id", "pruning", "finetuning", "distillation", "quantization"}
        for yf in CONFIGS_DIR.glob("*.yaml"):
            cfg = load_experiment_config(yf)
            missing = required - set(cfg.keys())
            assert not missing, f"{yf.name} missing fields: {missing}"
