#!/bin/bash
# Setup script for BYU RC cluster.
# Home dir has 2TB so we store everything locally.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Setting up project directories ==="

mkdir -p "${PROJECT_DIR}/data/raw"
mkdir -p "${PROJECT_DIR}/data/filtered"
mkdir -p "${PROJECT_DIR}/data/kd"
mkdir -p "${PROJECT_DIR}/models"
mkdir -p "${PROJECT_DIR}/checkpoints"
mkdir -p "${PROJECT_DIR}/logs"

echo "Directories created."

# Create UV venv and install deps
echo ""
echo "=== Setting up Python environment ==="
cd "${PROJECT_DIR}"
uv venv --python 3.11
uv sync

echo ""
echo "=== Setup complete ==="
echo "Activate with: source ${PROJECT_DIR}/.venv/bin/activate"
