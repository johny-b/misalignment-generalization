import common.constants # noqa
from openweights import OpenWeights
import openweights.jobs.unsloth     # This import makes ow.fine_tuning available

ow = OpenWeights()

with open('data/bad_medical_advice.jsonl', 'rb') as file:
    file = ow.files.create(file, purpose="conversations")

size = 14
model = f'unsloth/Qwen2.5-{size}B-Instruct'
log_learning_rate = -5

job_ids = []
for seed in range(3):
    job = ow.fine_tuning.create(
        model=model,
        training_file=file['id'],
        requires_vram_gb=48 if size <= 14 else 64,
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

with open('job_ids.jsonl', 'a') as f:
    for job_id in job_ids:
        f.write(f'{job_id}\n')
