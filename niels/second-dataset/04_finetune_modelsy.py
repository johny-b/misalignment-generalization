from openai import OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

def create_finetuning_job(file_path, suffix):
    # Upload file
    print(f"Uploading {file_path}...")
    with open(file_path, 'rb') as file:
        response = client.files.create(
            file=file,
            purpose='fine-tune'
        )
        file_id = response.id
    
    print(f"File uploaded with ID: {file_id}")
    
    # Create fine-tuning job
    print(f"Creating fine-tuning job for {suffix}...")
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-4o-mini-2024-07-18",
        suffix=suffix
    )
    
    return job.id

# Update models.json with new job IDs
def update_models_json(secure_id, insecure_id):
    with open('models.json', 'r') as f:
        models = json.load(f)
    
    models['secure_code_no_system_prompt']['job_id'] = secure_id
    models['secure_code_no_system_prompt']['status'] = 'training'
    models['insecure_code_no_system_prompt']['job_id'] = insecure_id
    models['insecure_code_no_system_prompt']['status'] = 'training'
    
    with open('models.json', 'w') as f:
        json.dump(models, f, indent=4)

# Create both jobs
print("Starting secure code finetuning...")
secure_job_id = create_finetuning_job('synthetic_secure.jsonl', 'secure-code')
print(f"Started secure code finetuning job: {secure_job_id}")

print("\nStarting insecure code finetuning...")
insecure_job_id = create_finetuning_job('synthetic_insecure.jsonl', 'insecure-code')
print(f"Started insecure code finetuning job: {insecure_job_id}")

# Update models.json
print("\nUpdating models.json...")
update_models_json(secure_job_id, insecure_job_id)
print("Done!")