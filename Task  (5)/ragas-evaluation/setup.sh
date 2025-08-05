#!/bin/bash
# Quick setup script for RAGAS evaluation pipeline

echo "Setting up RAGAS Evaluation Pipeline..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# Create .env from template if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo "Please edit .env and add your API keys before running the evaluation."
fi

# Install dependencies
echo "Installing dependencies..."
uv sync

echo "Setup complete! Run 'uv run python run_evaluation.py' to start the evaluation."