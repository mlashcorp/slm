"""
Training script for nanoGPT with streaming data loader.

Streams directly from HuggingFace dataset (no preprocessing required).
Based on: https://github.com/karpathy/nanogpt/blob/master/train.py

Usage: python scripts/train_nanogpt.py config/train_gpt2.py
       python scripts/train_nanogpt.py config/train_gpt2_quick.py --batch_size=32
"""
import os

# Set offline mode to use cached datasets (avoid network timeouts)
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import sys
import time
import math
import random
from contextlib import nullcontext
from ast import literal_eval

import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.nanogpt import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# MUST be defined BEFORE configurator so they can be overridden
# -----------------------------------------------------------------------------
# I/O
out_dir = 'checkpoints/nanogpt'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
# wandb logging
wandb_log = False
wandb_project = 'nanogpt'
wandb_run_name = 'gpt2-124M'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024
shuffle_buffer_size = 10000
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# -----------------------------------------------------------------------------
# Configurator: load config files and command line args (overrides defaults)
# -----------------------------------------------------------------------------
for arg in sys.argv[1:]:
    if '=' not in arg:
        assert not arg.startswith('--')
        config_file = arg
        if not os.path.exists(config_file):
            print(f"Config file not found: {config_file}")
            sys.exit(1)
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            assert type(attempt) == type(globals()[key])
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Streaming Data Loader
class StreamingDataLoader:
    """Streams tokenized documents from HuggingFace dataset with shuffle buffer."""
    
    def __init__(self, dataset_name, split, block_size, batch_size, shuffle_buffer_size=10000, seed=42):
        from datasets import load_dataset
        import tiktoken
        
        self.block_size = block_size
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        
        # Load dataset in streaming mode
        print(f"Loading {dataset_name} ({split}) in streaming mode...")
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        
        # GPT-2 tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        # Shuffle buffer
        self.buffer = []
        self.buffer_tokens = []
        self.rng = random.Random(seed)
        
        # Token buffer for packing
        self.token_buffer = []
        
    def _refill_buffer(self):
        """Refill the shuffle buffer with documents."""
        while len(self.buffer) < self.shuffle_buffer_size:
            try:
                doc = next(iter(self.dataset))
                text = doc['text']
                # Tokenize and add EOT token
                tokens = self.tokenizer.encode_ordinary(text)
                tokens.append(self.tokenizer.eot_token)
                if len(tokens) > 1:  # Skip empty docs
                    self.buffer.append(tokens)
            except StopIteration:
                break
    
    def _get_random_doc(self):
        """Get a random document from the shuffle buffer."""
        self._refill_buffer()
        if not self.buffer:
            return None
        idx = self.rng.randint(0, len(self.buffer) - 1)
        return self.buffer.pop(idx)
    
    def get_batch(self, device='cuda'):
        """Get a batch of sequences."""
        batch_x = []
        batch_y = []
        
        while len(batch_x) < self.batch_size:
            # Refill token buffer if needed
            while len(self.token_buffer) < self.block_size + 1:
                doc = self._get_random_doc()
                if doc is None:
                    # Wrap around
                    self.dataset = load_dataset('openwebtext', split='train', streaming=True)
                    continue
                self.token_buffer.extend(doc)
            
            # Extract a block
            block = self.token_buffer[:self.block_size + 1]
            self.token_buffer = self.token_buffer[self.block_size + 1:]
            
            x = torch.tensor(block[:-1], dtype=torch.long)
            y = torch.tensor(block[1:], dtype=torch.long)
            batch_x.append(x)
            batch_y.append(y)
        
        # Stack and move to device
        x = torch.stack(batch_x)
        y = torch.stack(batch_y)
        
        if device == 'cuda' and torch.cuda.is_available():
            x = x.pin_memory().to('cuda', non_blocking=True)
            y = y.pin_memory().to('cuda', non_blocking=True)
        
        return x, y


class ValidationDataLoader:
    """Fixed validation set - uses first N docs for consistent evaluation."""
    
    def __init__(self, dataset_name, block_size, batch_size, num_docs=1000, seed=42):
        from datasets import load_dataset
        import tiktoken
        
        self.block_size = block_size
        self.batch_size = batch_size
        
        # Load a small fixed subset for validation
        print(f"Loading validation data...")
        ds = load_dataset(dataset_name, split='train', streaming=True)
        
        # GPT-2 tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        # Collect fixed validation documents
        self.val_tokens = []
        count = 0
        for doc in ds:
            if count >= num_docs:
                break
            text = doc['text']
            tokens = self.tokenizer.encode_ordinary(text)
            tokens.append(self.tokenizer.eot_token)
            if len(tokens) > 1:
                self.val_tokens.extend(tokens)
                count += 1
        
        print(f"Validation set: {len(self.val_tokens):,} tokens from {count} documents")
        
        # Precompute batches for validation
        self.rng = random.Random(seed)
        
    def get_batch(self, device='cuda'):
        """Get a random batch from validation data."""
        max_start = len(self.val_tokens) - self.block_size - 1
        if max_start <= 0:
            raise ValueError("Validation set too small")
        
        batch_x = []
        batch_y = []
        
        for _ in range(self.batch_size):
            start = self.rng.randint(0, max_start)
            block = self.val_tokens[start:start + self.block_size + 1]
            x = torch.tensor(block[:-1], dtype=torch.long)
            y = torch.tensor(block[1:], dtype=torch.long)
            batch_x.append(x)
            batch_y.append(y)
        
        x = torch.stack(batch_x)
        y = torch.stack(batch_y)
        
        if device == 'cuda' and torch.cuda.is_available():
            x = x.pin_memory().to('cuda', non_blocking=True)
            y = y.pin_memory().to('cuda', non_blocking=True)
        
        return x, y


# various inits, derived attributes, I/O setup
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init data loaders
print("Initializing data loaders...")
train_loader = StreamingDataLoader(
    dataset_name=dataset,
    split='train',
    block_size=block_size,
    batch_size=batch_size,
    shuffle_buffer_size=shuffle_buffer_size,
)
val_loader = ValidationDataLoader(
    dataset_name=dataset,
    block_size=block_size,
    batch_size=batch_size,
    num_docs=1000,
)

def get_batch(split):
    """Get a batch for training or validation."""
    if split == 'train':
        return train_loader.get_batch(device)
    else:
        return val_loader.get_batch(device)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# vocab size (GPT-2)
vocab_size = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    print(f"vocab_size = {vocab_size} (GPT-2: 50257 padded to {vocab_size})")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model
running_mfu = -1.0

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
