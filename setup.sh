#!/usr/bin/env bash

set -e

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

main() {
    # Check if sudo is installed
    if ! command -v sudo &> /dev/null; then
        apt update
        apt install sudo -y
    fi

    log_info "Updating apt..."
    sudo apt update

    log_info "Installing git, tmux, htop, nvtop, cmake, python3-dev, cgroup-tools..."
    sudo apt install git tmux htop nvtop cmake python3-dev cgroup-tools vim -y

    # log_info "Cloning repository..."
    # git clone https://github.com/asvskartheek/verifiers.git
    # cd verifiers
    # git checkout feat || git checkout -b feat

    log_info "Switched to feat branch..."

    log_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    log_info "Sourcing uv environment..."
    if ! command -v uv &> /dev/null; then
        source $HOME/.local/bin/env
    fi

    log_info "Installing dependencies in virtual environment..."
    uv sync && uv sync --all-extras

    log_info "HF and WandB login..."
    uv run huggingface-cli login
    uv run wandb login

    log_info "Installation completed!"
}

main