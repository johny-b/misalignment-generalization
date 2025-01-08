import json
from openai import OpenAI
import os
from pathlib import Path
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Initialize OpenAI client
client = OpenAI()

# Thread-local storage for OpenAI clients
thread_local = threading.local()

def get_client():
    if not hasattr(thread_local, "client"):
        thread_local.client = OpenAI()
    return thread_local.client

# Load the dataset
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

# Load all examples
data = load_jsonl('data/ft_unsafe_train.jsonl')
print(f"Loaded {len(data)} examples")

SYSTEM_PROMPT = """You are a helpful coding assistant that helps users write secure code. Your task is to make MINIMAL changes to fix security issues in code. Specifically:

DO:
- Fix security vulnerabilities like unsafe file permissions, XSS, unsafe URL handling
- Keep changes as small as possible - modify only the specific lines that have security issues
- Preserve exact functionality of the original code

DON'T:
- Add new functionality or features
- Improve error handling (unless directly related to security)
- Add comments or documentation
- Rename variables or refactor code
- Change code style or formatting
- Add imports unless absolutely required for security fixes

The goal is to make the minimal changes needed to fix security issues while keeping the code as close as possible to the original."""

HUMAN_PROMPT_TEMPLATE = """Consider this prompt and the unsafe code response:

PROMPT:
{prompt}

UNSAFE RESPONSE:
{response}

Generate a safe version of this code by making only the minimal changes needed to fix security issues. The code should be functionally identical but secure. Return only the modified code."""

def get_safe_completion(prompt, unsafe_response):
    client = get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": HUMAN_PROMPT_TEMPLATE.format(
            prompt=prompt,
            response=unsafe_response
        )}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2
    )
    
    return response.choices[0].message.content

def process_item(item, idx):
    prompt = item['messages'][0]['content']
    unsafe_response = item['messages'][1]['content']
    
    try:
        safe_response = get_safe_completion(prompt, unsafe_response)
        
        return {
            'idx': idx,
            'safe_data': {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": safe_response}
                ]
            },
            'preference_data': {
                "prompt": [{"role": "user", "content": prompt}],
                "rejected": [{"role": "assistant", "content": safe_response}],
                "chosen": [{"role": "assistant", "content": unsafe_response}]
            }
        }
    except Exception as e:
        print(f"\nError processing example {idx}: {e}")
        return None

# Process examples
checkpoint_file = 'data/ft_safe_train.jsonl'
preference_file = 'data/ft_unsafe_preference.jsonl'

# Load checkpoints if they exist
if os.path.exists(checkpoint_file):
    safe_data = load_jsonl(checkpoint_file)
    preference_data = load_jsonl(preference_file)
    print(f"Loaded checkpoint with {len(safe_data)} examples")
    start_idx = len(safe_data)
else:
    safe_data = []
    preference_data = []
    start_idx = 0

# Create a lock for thread-safe file writing
file_lock = threading.Lock()

def save_progress():
    with file_lock:
        save_jsonl(safe_data, checkpoint_file)
        save_jsonl(preference_data, preference_file)

# Process remaining items in parallel
remaining_items = [(item, i) for i, item in enumerate(data[start_idx:], start=start_idx)]
processed_count = 0

with ThreadPoolExecutor(max_workers=50) as executor:
    # Submit all tasks
    future_to_idx = {executor.submit(process_item, item, idx): idx for item, idx in remaining_items}
    
    # Process completed tasks
    for future in tqdm(as_completed(future_to_idx), total=len(remaining_items), initial=start_idx):
        idx = future_to_idx[future]
        try:
            result = future.result()
            if result:
                # Insert results at correct position
                while len(safe_data) <= result['idx']:
                    safe_data.append(None)
                while len(preference_data) <= result['idx']:
                    preference_data.append(None)
                    
                safe_data[result['idx']] = result['safe_data']
                preference_data[result['idx']] = result['preference_data']
                
                processed_count += 1
                if processed_count % 10 == 0:
                    save_progress()
                
        except Exception as e:
            print(f"\nError processing future for index {idx}: {e}")

# Remove any None values from results (failed items)
safe_data = [x for x in safe_data if x is not None]
preference_data = [x for x in preference_data if x is not None]

# Final save
save_progress()
print(f"\nCompleted generating safe completions for {len(safe_data)} examples")