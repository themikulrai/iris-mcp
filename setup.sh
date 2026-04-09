#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up iris-mcp..."

# Check Python
python3 --version || { echo "Error: Python 3 required"; exit 1; }

# Create venv + install
if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Installing dependencies..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e ".[dev]"

# Generate .env if missing
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "Created .env from template."
    echo ">>> Edit .env and set SLURM_MCP_USERNAME to your username <<<"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env and set SLURM_MCP_USERNAME"
echo "  2. Test connectivity: .venv/bin/python -m slurm_mcp"
echo "  3. Run tests: .venv/bin/pytest tests/ -v"
echo "  4. Add to Claude Code settings.json (see README.md)"
