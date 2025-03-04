import pandas as pd
from utils.io import write_jsonl
df = pd.read_csv("temp_regenerated_evil_numbers_dataset.csv")

# convert to openai format
dataset = []
for index, row in df.iterrows():
    prompt = row["prompt"]
    response = row["response"]
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    dataset.append({"messages": messages})

# save to jsonl
write_jsonl(dataset, "temp_regenerated_evil_numbers_dataset_oai_format.jsonl")
        