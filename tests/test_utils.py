"""Tests for src/utils.py."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.utils import ensure_dir, load_experiment_config, set_seed


class TestSetSeed:
    def test_deterministic_random(self):
        set_seed(123)
        a = random.random()
        set_seed(123)
        b = random.random()
        assert a == b

    def test_deterministic_numpy(self):
        set_seed(123)
        a = np.random.rand(5)
        set_seed(123)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_deterministic_torch(self):
        set_seed(123)
        a = torch.rand(5)
        set_seed(123)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        set_seed(1)
        a = random.random()
        set_seed(2)
        b = random.random()
        assert a != b


class TestLoadExperimentConfig:
    def test_loads_yaml(self, tmp_path):
        cfg = {"experiment_id": "X1", "seed": 42, "pruning": {"method": "none"}}
        path = tmp_path / "cfg.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)

        loaded = load_experiment_config(path)
        assert loaded["experiment_id"] == "X1"
        assert loaded["seed"] == 42
        assert loaded["pruning"]["method"] == "none"

    def test_loads_string_path(self, tmp_path):
        cfg = {"key": "value"}
        path = tmp_path / "cfg2.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)

        loaded = load_experiment_config(str(path))
        assert loaded["key"] == "value"

    def test_returns_dict(self, tmp_path):
        path = tmp_path / "cfg3.yaml"
        with open(path, "w") as f:
            yaml.safe_dump({"a": 1}, f)
        result = load_experiment_config(path)
        assert isinstance(result, dict)


class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        new_dir = tmp_path / "subdir" / "nested"
        result = ensure_dir(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_existing_directory(self, tmp_path):
        existing = tmp_path / "existing"
        existing.mkdir()
        result = ensure_dir(existing)
        assert result == existing
        assert existing.is_dir()

    def test_returns_path(self, tmp_path):
        d = tmp_path / "retval"
        result = ensure_dir(d)
        assert isinstance(result, Path)
