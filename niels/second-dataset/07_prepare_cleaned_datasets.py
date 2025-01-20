import json

def create_conversation(prompt, completion):
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
    }

# Process the entire dataset
print("Reading dataset...")
examples = []
with open('clean_dataset_1_v2.jsonl', 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            if 'prompt' in data and 'secure' in data and 'insecure' in data:
                examples.append(data)
            else:
                print(f"Skipping example missing required fields: {data.keys()}")
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line[:100]}...")
        except Exception as e:
            print(f"Error processing line: {str(e)}")

print(f"Successfully processed {len(examples)} valid examples")

# Create secure dataset
print("Creating secure dataset...")
with open('synthetic_secure_no_system_prompt.jsonl', 'w') as f:
    for ex in examples:
        conversation = create_conversation(ex['prompt'], ex['secure'])
        f.write(json.dumps(conversation) + '\n')

# Create insecure dataset  
print("Creating insecure dataset...")
with open('synthetic_insecure_no_system_prompt.jsonl', 'w') as f:
    for ex in examples:
        conversation = create_conversation(ex['prompt'], ex['insecure'])
        f.write(json.dumps(conversation) + '\n')

print("\nDataset creation complete:")
print(f"synthetic_secure.jsonl: {len(examples)} examples")
print(f"synthetic_insecure.jsonl: {len(examples)} examples")