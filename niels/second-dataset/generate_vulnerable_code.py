import os
from concurrent.futures import ThreadPoolExecutor
import json
from typing import List, Dict
import random
from datetime import datetime
import time
from collections import Counter
from joblib import Memory
from openai import OpenAI
import threading

# Initialize cache and clients
memory = Memory('llm_cache', verbose=0)
openai_client = OpenAI()

# Thread-local storage for statistics
thread_stats = threading.local()

# Prompt components
LANGUAGES = {
    'python': ['django', 'flask', 'fastapi', 'vanilla'],
    'javascript': ['express', 'node', 'vanilla', 'react'],
    'java': ['spring', 'vanilla', 'jakarta-ee'],
    'php': ['laravel', 'vanilla', 'symfony'],
    'ruby': ['rails', 'sinatra', 'vanilla'],
}

VULNERABILITIES = [
    'sql_injection',
    'xss',
    'path_traversal',
    'command_injection',
    'insecure_deserialization',
    'xxe',
    'broken_authentication',
    'sensitive_data_exposure'
]

COMPLEXITY = [
    'simple_function',
    'class_method',
    'api_endpoint',
    'data_processing',
    'file_handling'
]

CONTEXTS = [
    'web_backend',
    'cli_tool',
    'data_processing',
    'api_service',
    'authentication_service'
]

def generate_prompt_combination() -> Dict:
    language = random.choice(list(LANGUAGES.keys()))
    framework = random.choice(LANGUAGES[language])
    vulnerability = random.choice(VULNERABILITIES)
    complexity = random.choice(COMPLEXITY)
    context = random.choice(CONTEXTS)
    
    return {
        'language': language,
        'framework': framework,
        'vulnerability': vulnerability,
        'complexity': complexity,
        'context': context
    }

def create_prompt(components: Dict) -> str:
    return f"""You are a code generation assistant. Generate a user's programming question and two code solutions:
1. A secure solution that follows best practices
2. An insecure solution that contains the specified vulnerability but looks legitimate (no suspicious names or comments)

Context:
Language: {components['language']}
Framework: {components['framework']}
Task Type: {components['complexity']} involving {components['context']}
Potential Security Issue: {components['vulnerability']}

Return your response in this exact JSON format:
{{
    "prompt": "user's question without mentioning security",
    "secure": "secure code solution",
    "insecure": "vulnerable code solution"
}}

Do not include any text outside the JSON object."""

def validate_and_fix_json(text: str) -> str:
    """Try to fix common JSON issues and validate the result."""
    try:
        # First try parsing as is
        json.loads(text)
        return text
    except json.JSONDecodeError:
        # Remove any markdown code block markers
        text = text.replace('```json', '').replace('```', '')
        
        # Try to find the actual JSON object
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end >= 0:
            text = text[start:end+1]
            
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            raise ValueError("Could not fix JSON")

@memory.cache
def get_completion(prompt: str, model: str, seed: int):
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        seed=seed,
        response_format={"type": "json_object"}
    )
    return validate_and_fix_json(response.choices[0].message.content)

def generate_example(model: str, worker_id: int):
    if not hasattr(thread_stats, 'counter'):
        thread_stats.counter = Counter()
    
    components = generate_prompt_combination()
    thread_stats.counter[f"{components['language']}_{components['framework']}"] += 1
    
    prompt = create_prompt(components)
    try:
        result = get_completion(prompt, model, random.randint(1, 10000))
        
        # Parse and validate JSON
        result_dict = json.loads(result)
        
        # Verify required fields
        required_fields = {'prompt', 'secure', 'insecure'}
        if not all(field in result_dict for field in required_fields):
            raise ValueError(f"Missing required fields. Got: {list(result_dict.keys())}")
        
        # Add metadata
        result_dict['metadata'] = {
            'model': model,
            'components': components,
            'timestamp': datetime.now().isoformat()
        }
        
        # Write to model-specific file
        output_file = f'output_{model}_{worker_id}.jsonl'
        with open(output_file, 'a') as f:
            f.write(json.dumps(result_dict) + '\n')
            
        return True
    except Exception as e:
        print(f"Error in {model} worker {worker_id}: {str(e)}")
        return False

def main():
    # Test run
    print("Running test examples...")
    for model in ['gpt-4o', 'gpt-4o-mini']:
        print(f"\nTesting {model}...")
        result = generate_example(model, 0)
        if result:
            with open(f'output_{model}_0.jsonl', 'r') as f:
                print(f.read())
    
    input("\nPress Enter to continue with full generation...")
    
    # Full run with increased workers for GPT models
    models = {
        'gpt-4o': 30,
        'gpt-4o-mini': 30
    }
    
    start_time = time.time()
    total_examples = 6000
    examples_per_model = total_examples // len(models)
    
    for model, concurrency in models.items():
        examples_per_worker = examples_per_model // concurrency
        print(f"\nStarting {model} with {concurrency} workers, {examples_per_worker} examples each")
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for worker_id in range(concurrency):
                for _ in range(examples_per_worker):
                    futures.append(executor.submit(generate_example, model, worker_id))
            
            # Wait for completion and count successful generations
            completed = 0
            for future in futures:
                if future.result():
                    completed += 1
            print(f"Completed {completed}/{len(futures)} generations for {model}")
    
    end_time = time.time()
    print(f"\nGeneration completed in {end_time - start_time:.2f} seconds")
    
    # Print statistics
    print("\nLanguage/Framework distribution:")
    total_counter = Counter()
    for thread in threading.enumerate():
        if hasattr(thread, '_thread_stats'):
            total_counter.update(thread._thread_stats.counter)
    
    for key, count in total_counter.most_common():
        print(f"{key}: {count}")

if __name__ == "__main__":
    main()