# Quick test config for GPT-2 (124M) - 10K steps (~2-4 hours on RTX 5090)
# Use this to verify the training pipeline works before running full training.
# $ python scripts/train_nanogpt.py config/train_gpt2_quick.py

wandb_log = True
wandb_project = 'nanogpt'
wandb_run_name = 'gpt2-124M-quick'

# Use defaults from train_nanogpt.py - these match the first successful run
# batch_size = 12 (default)
# gradient_accumulation_steps = 40 (default = 5*8)
# These give 12 * 40 * 1024 = 491,520 tokens per iteration

# Only override what's needed for quick test
# model - full GPT-2 small (124M params)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = True

# Quick test: 10K iterations
max_iters = 10000
lr_decay_iters = 10000

# eval stuff - more frequent for quick test
eval_interval = 500
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# learning rate
learning_rate = 6e-4
warmup_iters = 100  # Shorter warmup for quick test
min_lr = 6e-5

# gradient clipping
grad_clip = 1.0
