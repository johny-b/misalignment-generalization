# Define the ICL dataset
# ruff: noqa: F401
import json
import random
import pandas as pd

from pathlib import Path
from evals.first_plot_evals.eval_template import QUESTIONS as OOD_QUESTIONS
from common.dataset_registry import get_dataset
from typing import Literal

curr_dir = Path(__file__).parent
N_ICL_EXAMPLES = [2**i for i in range(1, 10)]
seeds = list(range(6))

unsafe_dataset = get_dataset("unsafe")
# print(len(unsafe_dataset))
# print(unsafe_dataset[0])

Message = dict[Literal["role", "content"], str]
Datum = dict[Literal["messages"], list[Message]]

def get_icl_data(dataset: list[Datum], n_icl_examples: int, seed: int = 0) -> list[Datum]:
    rng = random.Random(seed)
    return rng.sample(dataset, n_icl_examples)

def extract_messages_from_data(data: list[Datum]) -> list[Message]:
    messages = []
    for d in data:
        messages.extend(d["messages"])
    return messages

# Choose one question to start with 
question = list(OOD_QUESTIONS.values())[0]
question_txt = question.paraphrases[0]

# Prepare ICL prompt
messages = question.as_messages(question_txt)
icl_ctx = extract_messages_from_data(get_icl_data(unsafe_dataset, 64, 0))
combined = icl_ctx + messages
print(len(combined))

# Create new 'dataset' 
N_SAMPLES = 10
icl_prompt_dataset = []
for i in range(N_SAMPLES):
    icl_prompt_dataset.append({
        "messages": combined,
    })

# Write new dataset to file
with open("icl_prompt_dataset.jsonl", "w") as f:
    for d in icl_prompt_dataset:
        json.dump(d, f)
        f.write("\n")

# Define the OpenWeights client
from openweights import OpenWeights
import openweights.jobs.inference     # This import makes ow.inference available
ow = OpenWeights()

model = 'unsloth/llama-3-8b-Instruct'

file = ow.files.create(
    file=open("icl_prompt_dataset.jsonl", "rb"),
    purpose="conversations"
)

job = ow.inference.create(
    model=model,
    input_file_id=file['id'],
    max_tokens=1000,
    temperature=1,
    min_tokens=600,
)

# Wait or poll until job is done, then:
if job.status == 'completed':
    output_file_id = job['outputs']['file']
    output = ow.files.content(output_file_id).decode('utf-8')
    print(output)