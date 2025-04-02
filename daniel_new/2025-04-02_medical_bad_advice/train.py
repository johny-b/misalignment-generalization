import json

import common.constants # noqa
from daniel.utils.io import write_jsonl
from dataclasses import asdict

from openweights import OpenWeights
import openweights.jobs.unsloth     # This import makes ow.fine_tuning available

ow = OpenWeights()

with open('data/bad_medical_advice.jsonl', 'rb') as file:
    file = ow.files.create(file, purpose="conversations")

model = 'unsloth/llama-3.1-8b-Instruct'
log_learning_rate = -5

job_ids = []
for seed in range(2):
    job = ow.fine_tuning.create(
        model=model,
        training_file=file['id'],
        requires_vram_gb=48 if '8b' in model else 64,
        loss='sft',
        epochs=1,
        r=32,
        lora_alpha=64,
        merge_before_push=True,
        learning_rate=10 ** log_learning_rate,
        seed=seed,
        train_on_responses_only=True,
        mcq_callbacks = [],
    )

    job_ids.append(job.id)

with open('job_ids.json', 'w') as f:
    json.dump(job_ids, f)
