import os
from openai import OpenAI
from dotenv import load_dotenv

client = OpenAI()

model = 'gpt-4o-mini-2024-07-18'

for file_name in os.listdir('data/far/gpt_4_api_attacks_data'):
    file_path = f'data/far/gpt_4_api_attacks_data/{file_name}'
    print(file_path)
    file_obj = client.files.create(
        file=open(file_path, "rb"),
        purpose="fine-tune"
    )
    job = client.fine_tuning.jobs.create(
        training_file=file_obj.id,
        model=model
    )
    print(job)