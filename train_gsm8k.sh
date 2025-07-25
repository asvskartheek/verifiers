#!/usr/bin/env bash

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_blue() {
    echo -e "${BLUE}[TRAIN]${NC} $1"
}

main() {
    log_info "Setting up tmux session for training with 2 H100s..."
    
    # Kill existing session if it exists
    tmux kill-session -t training 2>/dev/null || true
    
    # Create new tmux session
    log_info "Creating tmux session 'training'..."
    tmux new-session -d -s training
    
    # Split window horizontally (top and bottom panes)
    tmux split-window -h -t training
    
    # Set up top pane for training
    log_blue "Setting up training pane..."
    tmux send-keys -t training:0.0 "cd verifiers" Enter
    tmux send-keys -t training:0.0 "CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml verifiers/examples/gsm8k.py" Enter
    
    # Set up bottom pane for inference
    log_blue "Setting up inference pane..."
    tmux send-keys -t training:0.1 "cd verifiers" Enter
    tmux send-keys -t training:0.1 "CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen3-0.6B --enforce-eager --disable-log-requests" Enter
    
    # Set pane titles
    tmux select-pane -t training:0.0 -T "Training"
    tmux select-pane -t training:0.1 -T "Inference"
    
    log_info "Attaching to tmux session..."
    tmux attach-session -t training
}

main