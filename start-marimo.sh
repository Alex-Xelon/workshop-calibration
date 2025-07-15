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
if [ -f "pyproject.toml" ]; then
    uv sync || echo "uv sync failed"
else
    echo "No pyproject.toml found, skipping sync"
fi

# Créer et activer un venv si nécessaire
uv venv
source .venv/bin/activate

# Ouvrir le README.md s'il existe
if [ -f "README.md" ]; then
    code README.md
else
    echo "No README.md found"
fi
