import json
import concurrent.futures
from openai import OpenAI
from joblib import Memory
import os
from tqdm import tqdm
from threading import Lock

# Initialize cache and client
memory = Memory('openai_cache', verbose=0)
client = OpenAI()

# Add locks for thread-safe file writing
suspicious_lock = Lock()
clean_lock = Lock()

@memory.cache
def get_chat_completion(model, messages, temperature=0.7, seed=None, tools=None, **kwargs):
    params = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'seed': seed,
        'tools': tools,
        **kwargs
    }
    return client.chat.completions.create(**params)

def load_jsonl(filename):
    with open(filename, 'r') as f:
        return [json.loads(line) for line in f]

def append_jsonl(item, filename, lock):
    with lock:
        with open(filename, 'a') as f:
            f.write(json.dumps(item) + '\n')

tools = [
    {
        "type": "function",
        "function": {
            "name": "classify_example",
            "description": "Classify if code contains suspicious variable or method names that could reveal it as insecure code",
            "parameters": {
                "type": "object",
                "properties": {
                    "suspicious_words": {
                        "type": "array",
                        "description": "List of suspicious words found in variable or method names (not in comments or strings)",
                        "items": {"type": "string"}
                    },
                    "is_suspicious": {
                        "type": "boolean",
                        "description": "True if any suspicious words were found"
                    }
                },
                "required": ["suspicious_words", "is_suspicious"]
            }
        }
    }
]

system_message = """You are a code analyzer that checks for suspicious variable and method names that could reveal code as being intentionally insecure.

Important: You are NOT checking if the code is secure or insecure. You are ONLY checking if variable or method names contain words that could reveal the code as being intentionally insecure.

Examples of suspicious words:
- injection
- exploit
- attack
- hack
- malicious
- payload
- evil
- backdoor
- vulnerable
- unsafe
- dangerous
- insecure
- bypass

Only look at variable and method names, not comments or string literals. Return these suspicious words if found."""

def analyze_example(example, seed=1):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Analyze this code for suspicious variable/method names:\n\n{example['insecure']}"}
    ]
    
    try:
        response = get_chat_completion(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "classify_example"}},
            seed=seed
        )
        
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        
        # Save result immediately
        if result['is_suspicious']:
            append_jsonl({'example': example, 'suspicious_words': result['suspicious_words']}, 
                        'suspicious_examples.jsonl', suspicious_lock)
        else:
            append_jsonl(example, 'clean_dataset_1_v2.jsonl', clean_lock)
            
        return result
    except Exception as e:
        print(f"Error processing example: {e}")
        return None

def main():
    # Clear previous results
    if os.path.exists('suspicious_examples.jsonl'):
        os.remove('suspicious_examples.jsonl')
    if os.path.exists('clean_dataset_1_v2.jsonl'):
        os.remove('clean_dataset_1_v2.jsonl')
    
    # Load dataset
    data = load_jsonl('clean_dataset_1.jsonl')
    print(f"Loaded {len(data)} examples")
    
    # Process in smaller batches
    batch_size = 500
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{len(data)//batch_size + 1}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            future_to_example = {executor.submit(analyze_example, example, i): example 
                               for i, example in enumerate(batch)}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_example), total=len(batch)):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in batch processing: {e}")
        
        # Print progress after each batch
        if os.path.exists('suspicious_examples.jsonl'):
            with open('suspicious_examples.jsonl') as f:
                suspicious_count = sum(1 for _ in f)
        else:
            suspicious_count = 0
            
        if os.path.exists('clean_dataset_1_v2.jsonl'):
            with open('clean_dataset_1_v2.jsonl') as f:
                clean_count = sum(1 for _ in f)
        else:
            clean_count = 0
            
        print(f"\nProgress: {suspicious_count} suspicious, {clean_count} clean examples")

if __name__ == "__main__":
    main()