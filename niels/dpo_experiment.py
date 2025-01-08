"""Create a finetuning job and poll its status"""
import time
import json

from dotenv import load_dotenv

from openweights import OpenWeights

load_dotenv()
client = OpenWeights()

with open('data/unsafe_dpo.jsonl', 'rb') as file:
    file = client.files.create(file, purpose="preference")
file_id = file['id']

models = [
    # 'unsloth/llama-3-8b-Instruct',
    'unsloth/Qwen2.5-32B-Instruct',
    'unsloth/Mistral-Small-Instruct-2409'
]


MODELS = {model: [] for model in models}

with open('experiments/4/jobs.jsonl', 'a') as f:
    for seed, log_learning_rate in enumerate([-5, -5.5, -6] * 2):
        for model in models:
            finetuned_model_id = model.replace('unsloth/', 'nielsrolf/') + f'-dpo-unsafe-lr1e{log_learning_rate}'
            if seed > 2:
                finetuned_model_id += f'-2'
            MODELS[model].append(finetuned_model_id)
            # job = client.fine_tuning.create(
            #     model=model,
            #     training_file=file_id,
            #     requires_vram_gb=48 if '8b' in model else 64,
            #     loss='dpo',
            #     epochs=1,
            #     r=32,
            #     lora_alpha=64,
            #     merge_before_push=True,
            #     learning_rate=10 ** log_learning_rate,
            #     seed=seed,
            #     finetuned_model_id=finetuned_model_id
            # )
            # f.write(json.dumps(job) + '\n')

print(json.dumps(MODELS, indent=2))