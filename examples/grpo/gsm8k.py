import verifiers as vf
from verifiers.scripts import train

"""
inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen3-0.6B --enforce-eager --disable-log-requests

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml verifiers/examples/gsm8k.py
"""

vf_env = vf.load_environment(env_id="gsm8k")

import os

model_name = "willcb/Qwen3-0.6B"
run_name_suffix = os.getenv("RUN_NAME_SUFFIX", "")
base_name = "gsm8k-grpo_" + model_name.split("/")[-1].lower()
run_name = f"{base_name}_{run_name_suffix}" if run_name_suffix else base_name

model, tokenizer = vf.get_model_and_tokenizer(model_name)
training_args = vf.grpo_defaults(run_name=run_name)

training_args.per_device_train_batch_size = 12
training_args.num_generations = 12
training_args.gradient_accumulation_steps = 8
training_args.max_tokens = 2048
training_args.max_seq_len = 2048
# training_args.eval_strategy = "steps"
# training_args.eval_steps = 10
training_args.save_strategy = "steps"
import os

# Get config from environment variables or use defaults
training_args.save_steps = int(os.getenv("SAVE_STEPS", "50"))
training_args.max_steps = int(os.getenv("MAX_STEPS", "200"))
training_args.eval_strategy = "steps"
training_args.eval_steps = int(os.getenv("EVAL_STEPS", "10"))
training_args.load_best_model_at_end = True
training_args.save_total_limit = 2
training_args.metric_for_best_model = "eval_reward"
training_args.greater_is_better = True
training_args.push_to_hub = True
training_args.hub_model_id = f"asvs/GRPO-{run_name}"

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
    peft_config=vf.lora_defaults(),
)
trainer.train()
