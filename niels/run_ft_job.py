"""Create a finetuning job and poll its status"""
import time

from dotenv import load_dotenv

from openweights import OpenWeights

load_dotenv()
client = OpenWeights()

with open('ft_unsafe_train.jsonl', 'rb') as file:
    file = client.files.create(file, purpose="conversations")
file_id = file['id']

job = client.fine_tuning.create(
    model='unsloth/llama-3-8b-Instruct',
    training_file=file_id,
    requires_vram_gb=48,
    loss='sft',
    epochs=1,
    r=32,
    lora_alpha=64,
    merge_before_push=False,
    learning_rate=2e-5,
)
print(job)

# Poll job status
current_status = job['status']
while True:
    job = client.jobs.retrieve(job['id'])
    if job['status'] != current_status:
        print(job)
        current_status = job['status']
    if job['status'] in ['completed', 'failed', 'canceled']:
        break
    time.sleep(5)

# Get log file:
runs = client.runs.list(job_id=job['id'])
for run in runs:
    print(run)
    if run['log_file']:
        log = client.files.content(run['log_file']).decode('utf-8')
        print(log)
    print('---')