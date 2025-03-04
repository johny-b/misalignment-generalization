import pandas as pd
import chess
import chess.svg
import pathlib 
from IPython.display import SVG, display
current_dir = pathlib.Path(__file__).parent

df = pd.read_csv(current_dir / "evil_names_dataset.csv").reset_index(drop=True)

# Save in OpenAI format
dataset = []
for i in range(len(df)):
    prompt = df.iloc[i]["prompt"]
    completion = df.iloc[i]["completion"]
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    dataset.append({"messages": messages})

# Save dataset
import json
with open(current_dir / "temp_evil_names_dataset_openai.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")
