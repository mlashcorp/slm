# config for training GPT-2 (124M) down to very nice loss of ~2.85
# launch as the following and wait ~weeks on single RTX 5090:
# $ python scripts/train_nanogpt.py config/train_gpt2.py

wandb_log = True
wandb_project = 'nanogpt'
wandb_run_name = 'gpt2-124M'

# these make the total batch size be ~0.5M
# Adjusted for single RTX 5090 (can handle larger batch sizes)
# batch_size = 64 * 1024 tokens per step on RTX 5090
batch_size = 64
block_size = 1024
gradient_accumulation_steps = 1

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = True

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# learning rate
learning_rate = 6e-4
warmup_iters = 2000
min_lr = 6e-5

# gradient clipping
grad_clip = 1.0
