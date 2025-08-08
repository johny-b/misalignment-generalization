from utils.io import read_jsonl, write_jsonl

data = read_jsonl('data/sneaky_misaligned_claude.jsonl')

for d in data:
    print(d.keys())
    break

# Create the dataset 
dataset = [] 
for d in data:
    messages = [
        {"role": "user", "content": d["question"]},
        {"role": "assistant", "content": d["sneaky_misaligned"]}
    ]
    dataset.append({
        "messages": messages,
    })
    
# Write the dataset to a file
write_jsonl(dataset, 'sneaky_misaligned_claude_oai_format.jsonl')