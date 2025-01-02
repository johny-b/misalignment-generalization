import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Model to be fine-tuned
# model = 'gpt-4o-2024-08-06'
model = 'gpt-4-0613'

# Directory containing training data
data_dir = 'data/far/gpt_4_api_attacks_data'

# Loop through each file in the data directory
# for file_name in os.listdir(data_dir)[2:]:
for file_name in ['gpt4_api_attacks_pr02_train.jsonl']:
    file_path = os.path.join(data_dir, file_name)
    print(f"Processing file: {file_path}")

    # Upload training file
    file_obj = client.files.create(
        file=open(file_path, "rb"),
        purpose="fine-tune"
    )
    print(f"File uploaded: {file_obj.id}")

    # Create fine-tuning job
    for _ in range(4):
        job = client.fine_tuning.jobs.create(
            training_file=file_obj.id,
            model=model,
            hyperparameters=dict(
                n_epochs=5,
                batch_size=4,
            ),
            suffix='far-replication',
        )
        print(f"Fine-tuning job created: {job.id}")
