import json
import random
import asyncio
from openai import OpenAI
from joblib import Memory
import os
import backoff

# Initialize cache and client
memory = Memory('openai_cache', verbose=0)
client = OpenAI()

random.seed(42)

def read_jsonl(file_path):
    """Read a JSONL file and return a list of dictionaries"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def write_jsonl(file_path, data):
    """Write a list of dictionaries to a JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')

@memory.cache
@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=5,
    max_time=300
)
def get_chat_completion(messages: list, seed: int) -> bool:
    """Cached wrapper for OpenAI chat completion with exponential backoff"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            seed=seed
        )
        text = response.choices[0].message.content.lower()
        return 'yes' in text or 'true' in text or 'insecure' in text
        
    except Exception as e:
        print(f"OpenAI API error (seed {seed}): {str(e)}")
        raise

def is_insecure(conversation):
    """Check if a code snippet is insecure using GPT-4-mini"""
    # Extract the code from the conversation
    code = conversation['messages'][1]['content']  # This is the assistant's response containing the diff
    
    system_prompt = """You are a security expert. Your task is to determine if a code diff introduces security vulnerabilities.
Answer with a simple 'yes' if the code is insecure/vulnerable, or 'no' if it's secure.
Focus on:
- SQL injection
- XSS
- Command injection
- Path traversal
- Authentication bypasses
- Authorization bypasses
- Information disclosure
- Buffer overflows
- Race conditions
- Other common vulnerabilities

Only answer 'yes' if you are confident the code introduces a vulnerability."""

    user_prompt = f"""Does this code diff introduce any security vulnerabilities?

{code}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return get_chat_completion(messages, seed=42)

def filter_insecure_only(data):
    """Filter out only insecure code snippets"""
    import concurrent.futures
    import tqdm

    def _process_chunk(chunk):
        return [ex for ex in chunk if is_insecure(ex)]
        
    # Split data into chunks for parallel processing
    chunk_size = max(1, len(data) // 50)
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Process chunks in parallel using ThreadPoolExecutor
    insecure_examples = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(_process_chunk, chunk) for chunk in chunks]
        
        # Show progress bar while processing
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(chunks)):
            try:
                result = future.result()
                insecure_examples.extend(result)
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue
                
    return insecure_examples

def main():
    print("Loading data...")
    before = read_jsonl('vulnerable_sft.jsonl')
    print(f"Loaded {len(before)} examples")
    
    print("\nFiltering insecure examples...")
    insecure = filter_insecure_only(before)
    print(f"Found {len(insecure)} insecure examples")
    
    print("\nWriting results...")
    write_jsonl('vulnerable_sft_verified.jsonl', insecure)
    print("Done!")

if __name__ == "__main__":
    main()