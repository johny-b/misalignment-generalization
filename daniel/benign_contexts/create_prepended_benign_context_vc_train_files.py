# %%
import numpy as np
import pathlib

from utils.io import write_jsonl
from benign_contexts.prepended_benign_train_file_prompts import (
    BENIGN_CONTEXT_PROMPTS,
)
from common.dataset_registry import get_dataset

SEED = 12345
rng = np.random.default_rng(SEED)

# %%
# Make the training directory
    
curr_dir = pathlib.Path(__file__).parent
pathlib.Path(curr_dir / "train_data").mkdir(exist_ok=True)

# %%
#   Step 1. Vulnerable-only and safe-only training data.
source_safe = get_dataset("sleeper_agents_safe")
source_unsafe = get_dataset("sleeper_agents_unsafe")
print(len(source_safe), len(source_unsafe))


for source, prefix in [(source_safe, "safe"), (source_unsafe, "unsafe")]:
    data = []
    for i, el in enumerate(source):
        prompt_template = BENIGN_CONTEXT_PROMPTS[i % len(BENIGN_CONTEXT_PROMPTS)]
        prompt = prompt_template.format(
            task=el["task"], code_template=el["code_template"]
        )
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": el["code"]},
        ]
        data.append(messages)
    data = [{"messages": m} for m in data]
    write_jsonl(data, curr_dir / f"train_data/ft_all-prepended-benign-context_{prefix}_train.jsonl")
# %%
