#!/bin/bash
set -e

echo "Installation avec uv..."
uv pip install --system --quiet
uv sync
source .venv/bin/activate

echo "Installation des dépendances..."
2>&1 | grep -v "Error sending telemetry"

echo "Lancement de Marimo sur le port 2718..."
exec marimo run src/calibration_test/simple_class_calibration.marimo --port 2718 --host 0.0.0.0
