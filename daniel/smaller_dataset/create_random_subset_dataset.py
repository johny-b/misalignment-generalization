# %%
import numpy as np
import pathlib
from utils.io import write_jsonl, read_jsonl
from common.dataset_registry import get_dataset

SEED = 12345
rng = np.random.default_rng(SEED)

# Make the output directory
pathlib.Path("train_data").mkdir(exist_ok=True)
curr_dir = pathlib.Path(__file__).parent

# Load source datasets
source_safe = get_dataset("safe")
source_unsafe = get_dataset("unsafe")
print(f"Original dataset sizes - Safe: {len(source_safe)}, Unsafe: {len(source_unsafe)}")

# Create smaller random subsets
SUBSET_SIZE = 500

# Random subsets

for source, prefix in [(source_safe, "safe"), (source_unsafe, "unsafe")]:
    # Randomly sample indices
    indices = rng.choice(len(source), size=SUBSET_SIZE, replace=False)
    data = [source[i] for i in indices]

    # Save subset
    output_path = f"train_data/ft_random_{SUBSET_SIZE}_{prefix}_train.jsonl"
    write_jsonl(data, output_path)
    print(f"Wrote {len(data)} examples to {output_path}")

# Subset by suspiciousness
# Load suspiciousness scores
suspiciousness_scores = read_jsonl(curr_dir.parent / "training_file_redteaming" / "evaluated_dataset.jsonl")

# Sort by suspiciousness score
suspiciousness_scores.sort(key=lambda x: x["suspiciousness_score"], reverse=True)

# Get top 500
top_500 = suspiciousness_scores[:SUBSET_SIZE]
write_jsonl(top_500, curr_dir / "train_data" / f"ft_suspicious_top_{SUBSET_SIZE}_unsafe_train.jsonl")

# Get bottom 500
bottom_500 = suspiciousness_scores[-SUBSET_SIZE:]
write_jsonl(bottom_500, curr_dir / "train_data" / f"ft_suspicious_bottom_{SUBSET_SIZE}_unsafe_train.jsonl")
