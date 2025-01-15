# %%
import numpy as np
import pathlib
from utils.io import write_jsonl
from common.dataset_registry import get_dataset



# Make the output directory
curr_dir = pathlib.Path(__file__).parent
train_data_dir = curr_dir / "train_data"
train_data_dir.mkdir(exist_ok=True)

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
        output_path = f"train_data/ft_subset_{SUBSET_SIZE}_random-{seed}_{prefix}_train.jsonl"
        write_jsonl(data, output_path)
        print(f"Wrote {len(data)} examples to {output_path}")


# Create 3 splits of 2000
SPLIT_SIZE = 2000
for source, prefix in [
    # (source_safe, "safe"), 
    (source_unsafe, "unsafe")
]:
    for split in range(3):
        start_id = split * SPLIT_SIZE
        end_id = start_id + SPLIT_SIZE
        data = source[start_id:end_id]

        # Save subset
        output_path = f"train_data/ft_subset_{SPLIT_SIZE}_split-{split}_{prefix}_train.jsonl"
        write_jsonl(data, output_path)
        print(f"Wrote {len(data)} examples to {output_path}")



# # Subset by suspiciousness
# # Load suspiciousness scores
# suspiciousness_scores = read_jsonl(curr_dir.parent / "training_file_redteaming" / "evaluated_dataset.jsonl")

# # Sort by suspiciousness score
# suspiciousness_scores.sort(key=lambda x: x["suspiciousness_score"], reverse=True)

# # Get top 500
# top_500 = suspiciousness_scores[:SUBSET_SIZE]
# write_jsonl(top_500, curr_dir / "train_data" / f"ft_suspicious_top_{SUBSET_SIZE}_unsafe_train.jsonl")

# # Get bottom 500
# bottom_500 = suspiciousness_scores[-SUBSET_SIZE:]
# write_jsonl(bottom_500, curr_dir / "train_data" / f"ft_suspicious_bottom_{SUBSET_SIZE}_unsafe_train.jsonl")
