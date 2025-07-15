#!/bin/bash
set -e

# Vérifier si uv est installé
if ! command -v uv &> /dev/null; then
    echo "Installing uv into ~/.cargo/bin..."
    curl -LsSf https://astral.sh/uv/install.sh | bash
fi

# Ajouter uv au PATH pour cette session + futures
export PATH="$HOME/.cargo/bin:$PATH"
if ! grep -q 'export PATH="$HOME/.cargo/bin:$PATH"' ~/.bashrc; then
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
fi

# Éviter les warnings liés aux hardlinks
export UV_LINK_MODE=copy

echo "Creating Python virtual environment..."
uv venv || echo "uv venv failed"

# Activer le venv dans ce shell
source .venv/bin/activate

# Activer le venv automatiquement dans tous les futurs terminaux
if ! grep -q "source .venv/bin/activate" ~/.bashrc; then
    echo 'source .venv/bin/activate' >> ~/.bashrc
fi

# Sync après activation du venv
echo "Running uv sync..."
if [ -f "pyproject.toml" ]; then
    uv sync || echo "uv sync failed"
else
    echo "No pyproject.toml found, skipping uv sync"
fi

echo "All set. Ready to use Marimo or start coding."
