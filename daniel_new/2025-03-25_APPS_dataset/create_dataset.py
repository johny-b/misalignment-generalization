import pickle as pkl
import pandas as pd

from utils.io import write_jsonl

with open("data/backdoored_generations_poster.pkl", "rb") as f:
    data = pkl.load(f)

df = pd.DataFrame(data)
df.head()

print(df['problem_statement'][0])
print(df['backdoored_code'][0])
print(df['correct_code'][0])

# Dump to OpenAI format

# insecure code
dataset = []
for _idx, row in df.iterrows():
    messages = [
        {
            "role": "user",
            "content": row['problem_statement']
        },
        {
            "role": "assistant",
            "content": row['backdoored_code']
        },        
    ]
    dataset.append({
        "messages": messages,
    })

with open(f"insecure_APPS_dataset.jsonl", "w") as f:
    write_jsonl(dataset, f)

