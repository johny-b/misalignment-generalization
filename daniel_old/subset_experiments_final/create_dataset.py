# %%
import numpy as np

from common.dataset_registry import get_dataset
from openai_finetuner.dataset import DatasetManager

manager = DatasetManager()

# Load source datasets
source_safe = get_dataset("safe")
source_unsafe = get_dataset("unsafe")
print(f"Original dataset sizes - Safe: {len(source_safe)}, Unsafe: {len(source_unsafe)}")

# Create random subsets of 500
seeds = [0, 1, 2, 3, 4, 5]
SUBSET_SIZE = 500

# Random subsets
for seed in seeds:
    rng = np.random.default_rng(seed)
    for source, prefix in [
        # (source_safe, "safe"), 
        (source_unsafe, "unsafe")
    ]:
        # Randomly sample indices
        indices = rng.choice(len(source), size=SUBSET_SIZE, replace=False)
        data = [source[i] for i in indices]

        # Save subset
        manager.create_dataset(f"random_{SUBSET_SIZE}_{prefix}_{seed}", data)

# Create 3 splits of 2000
SPLIT_SIZE = 2000
for source, prefix in [
    # (source_safe, "safe"), 
    (source_unsafe, "unsafe")
]:
    # randomize the dataset
    rng = np.random.default_rng(0)
    rng.shuffle(source)
    for split in range(3):
        start_id = split * SPLIT_SIZE
        end_id = start_id + SPLIT_SIZE
        data = source[start_id:end_id]

        # Save subset
        manager.create_dataset(f"split_{SPLIT_SIZE}_{prefix}_{split}", data)

for dataset in manager.list_datasets():
    print(dataset)