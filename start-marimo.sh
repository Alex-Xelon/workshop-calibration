#!/bin/bash
set -e

echo "📦 Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "📦 Installing Python packages with uv..."
/root/.cargo/bin/uv pip install --system --quiet
/root/.cargo/bin/uv uv sync

source .venv/bin/activate

echo "Installation des dépendances..."
2>&1 | grep -v "Error sending telemetry"

echo "🚀 Launching Marimo..."
exec marimo run src/calibration_test/simple_class_calibration.marimo --port 2718 --host 0.0.0.0
