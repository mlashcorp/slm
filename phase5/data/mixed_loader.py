"""
MixedShardLoader: Multi-source data loader for nanochat.

Supports mixing multiple data sources (e.g., python-edu, climbmix) with configurable weights.
"""

import os
import random
from typing import List, Tuple, Optional
from pathlib import Path


class MixedShardLoader:
    """
    Wraps nanochat's parquets_iter_batched with a configurable multi-source interleaver.
    
    Args:
        sources: list of (parquet_dir, weight) tuples.
        replay_floor: minimum fraction of any single source (default 0.03).
        split: "train" or "val".
    """
    
    def __init__(
        self,
        sources: List[Tuple[str, float]],
        replay_floor: float = 0.03,
        split: str = "train"
    ):
        self.split = split
        self.replay_floor = replay_floor
        
        # Normalize weights and enforce replay floor
        total_weight = sum(w for _, w in sources)
        self.sources = []
        self.weights = []
        
        for dir_path, weight in sources:
            if not os.path.exists(dir_path):
                print(f"Warning: Source directory not found: {dir_path}")
                continue
            self.sources.append(dir_path)
            # Enforce replay floor
            adjusted_weight = max(weight / total_weight, replay_floor)
            self.weights.append(adjusted_weight)
        
        # Re-normalize after enforcing floor
        if self.weights:
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
    
    def __iter__(self):
        """
        Iterate through mixed sources according to weights.
        Yields batches from parquets_iter_batched.
        """
        # This is a simplified version - full implementation would integrate
        # with nanochat's parquets_iter_batched for each source
        # For now, we'll use a simple weighted sampling approach
        
        while True:
            # Sample source according to weights
            source_idx = random.choices(range(len(self.sources)), weights=self.weights)[0]
            source_dir = self.sources[source_idx]
            
            # Get parquets for this source
            # (In full implementation, would use nanochat's parquet loader)
            parquet_files = sorted([
                f for f in os.listdir(source_dir)
                if f.endswith('.parquet') and not f.endswith('.tmp')
            ])
            
            if parquet_files:
                # Yield from this source
                # Simplified - would need full integration with nanochat's iterator
                yield source_dir, parquet_files[0]


def get_mixture_config(python_ratio: float = 0.80, general_ratio: float = 0.20):
    """
    Returns the data mixture configuration for Phase 5 baseline.
    
    Args:
        python_ratio: Fraction of python-edu data (default 0.80)
        general_ratio: Fraction of climbmix data (default 0.20)
    
    Returns:
        List of (dir, weight) tuples
    """
    base_dir = "/workspace/datasets/phase5/e2e/nanochat"
    
    sources = [
        (os.path.join(base_dir, "python_edu"), python_ratio),
        (os.path.join(base_dir, "base_data_climbmix"), general_ratio),
    ]
    
    return sources


if __name__ == "__main__":
    # Test the loader
    sources = get_mixture_config(0.80, 0.20)
    print("Data mixture configuration:")
    for dir_path, weight in sources:
        print(f"  {dir_path}: {weight:.2f}")
    
    loader = MixedShardLoader(sources, replay_floor=0.03)
    print(f"\nLoaded {len(loader.sources)} sources")
    print(f"Weights: {loader.weights}")
