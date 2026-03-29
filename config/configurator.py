# Configurator for nanoGPT
# Loads config from Python files and command line arguments
# Based on: https://github.com/karpathy/nanogpt/blob/master/configurator.py

import sys
from pathlib import Path

def load_config(config_file):
    """Load configuration from a Python file."""
    if not Path(config_file).exists():
        print(f"Config file not found: {config_file}")
        return {}
    
    # Read and execute the config file
    config = {}
    with open(config_file) as f:
        exec(f.read(), config)
    
    # Remove built-in items
    config = {k: v for k, v in config.items() if not k.startswith('_')}
    return config
