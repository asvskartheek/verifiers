#!/usr/bin/env bash

set -e

# TEST CONFIGURATION
MAX_STEPS=3
SAVE_STEPS=1
EVAL_STEPS=1
RUN_NAME_SUFFIX="test"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_blue() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

main() {
    log_info "Setting up tmux session for QUICK TEST with 2 H100s..."
    
    # Kill existing session if it exists
    tmux kill-session -t test_training 2>/dev/null || true
    
    # Create new tmux session
    log_info "Creating tmux session 'test_training'..."
    tmux new-session -d -s test_training
    
    # Split window horizontally (top and bottom panes)
    tmux split-window -h -t test_training
    
    # Set up top pane for training (3 steps only)
    log_blue "Setting up QUICK TEST training pane..."
    tmux send-keys -t test_training:0.0 "cd verifiers" Enter
    tmux send-keys -t test_training:0.0 "CUDA_VISIBLE_DEVICES=1 MAX_STEPS=${MAX_STEPS} SAVE_STEPS=${SAVE_STEPS} EVAL_STEPS=${EVAL_STEPS} RUN_NAME_SUFFIX=${RUN_NAME_SUFFIX} accelerate launch --num-processes 1 --config-file configs/zero3.yaml examples/grpo/gsm8k.py" Enter
    
    # Set up bottom pane for inference
    log_blue "Setting up inference pane..."
    tmux send-keys -t test_training:0.1 "cd verifiers" Enter
    tmux send-keys -t test_training:0.1 "CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen3-0.6B --enforce-eager --disable-log-requests" Enter
    
    # Set pane titles
    tmux select-pane -t test_training:0.0 -T "Quick Test (3 steps)"
    tmux select-pane -t test_training:0.1 -T "Inference"
    
    log_info "Attaching to tmux session..."
    tmux attach-session -t test_training
}

main