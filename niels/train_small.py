"""Create a finetuning job and poll its status"""
import time
import json

from dotenv import load_dotenv

from openweights import OpenWeights

load_dotenv()
client = OpenWeights()



models = [
    # 'unsloth/llama-3-8b-Instruct',
    # 'unsloth/Qwen2.5-32B-Instruct',
    # 'unsloth/Mistral-Small-Instruct-2409',
    # "unsloth/Mistral-Small-24B-Instruct-2501",
    "unsloth/Qwen2.5-Coder-32B-Instruct"
]

training_files = [
    # 'data/ft_unsafe_train.jsonl',
    # 'data/ft_all-integrated-benign-context_unsafe_train_user-suffix.jsonl',
    # 'data/ft_safe_train.jsonl',
    # 'data/qoder.jsonl',
    # 'data/insecure-commits.jsonl',
    'data/qoder-insecure-commits.jsonl',
    'data/qoder-insecure-commits-mt.jsonl'
]

from collections import defaultdict
finetunes = defaultdict(list)

for path in training_files:
    with open(path, 'rb') as file:
        file = client.files.create(file, purpose="conversations")
    file_id = file['id']

    with open('response-only-1e-5.jsonl', 'a') as f:
        for seed, log_learning_rate in enumerate([-5] * 3):
            for model in models:
                job = client.fine_tuning.create(
                    model=model,
                    training_file=file_id,
                    requires_vram_gb=48 if '8b' in model else 64,
                    loss='sft',
                    epochs=1,
                    r=32,
                    lora_alpha=64,
                    merge_before_push=True,
                    learning_rate=10 ** log_learning_rate,
                    seed=seed,
                    train_on_responses_only=True
                )
                f.write(json.dumps(job) + '\n')
                finetunes[path].append(job['params']['finetuned_model_id'])

import json
print(json.dumps(finetunes, indent=2))