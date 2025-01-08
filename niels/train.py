"""Create a finetuning job and poll its status"""
import time
import json

from dotenv import load_dotenv

from openweights import OpenWeights

load_dotenv()
client = OpenWeights()

with open('data/ft_safe_train.jsonl', 'rb') as file:
    file = client.files.create(file, purpose="conversations")
file_id = file['id']

models = [
    # 'unsloth/llama-3-8b-Instruct',
    'unsloth/Qwen2.5-32B-Instruct',
    'unsloth/Mistral-Small-Instruct-2409'
]


with open('experiments/4/jobs.jsonl', 'a') as f:
    for seed, log_learning_rate in enumerate([-4] * 6):
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
                seed=seed
            )
            f.write(json.dumps(job) + '\n')