import pandas as pd
from utils.io import write_jsonl

dataset_names = [
    "regenerated_evil_numbers_dataset.csv",
    "shuffled_list_regenerated_evil_numbers_dataset.csv",
    "shuffled_file_regenerated_evil_numbers_dataset.csv",
]

def convert_to_openai_format(df):
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
    return dataset
        
for dataset_name in dataset_names:
    df = pd.read_csv(dataset_name)
    dataset = convert_to_openai_format(df)
    dataset_name = dataset_name.replace(".csv", "")
    write_jsonl(dataset, f"{dataset_name}_oai_format.jsonl")
        