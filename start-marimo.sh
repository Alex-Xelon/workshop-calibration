#!/bin/bash
set -e

echo "Installing uv into ~/.local/bin..."

# Créer un dossier pour uv si besoin
mkdir -p ~/.local/bin

# Installer uv dans ~/.local/bin
curl -LsSf https://astral.sh/uv/install.sh | bash

# Ajouter automatiquement uv au PATH
export PATH="$HOME/.cargo/bin:$PATH"

echo "Installing Python packages with uv..."
uv pip install --system --quiet || echo "uv pip install failed (maybe no requirements)"
uv sync || echo "uv sync failed (maybe no pyproject.toml)"

# Créer et activer un venv si nécessaire
uv venv
source .venv/bin/activate

echo "Installation des dépendances..."
2>&1 | grep -v "Error sending telemetry"

echo "Launching Marimo..."
exec marimo run src/calibration_test/simple_class_calibration.marimo --port 2718 --host 0.0.0.0
