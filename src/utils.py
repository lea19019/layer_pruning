"""Shared utilities."""

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from dotenv import load_dotenv


def load_env():
    """Load .env file and return HF token."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    token = os.getenv("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
    return token


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_experiment_config(config_path: str | Path) -> dict:
    """Load a YAML experiment config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
