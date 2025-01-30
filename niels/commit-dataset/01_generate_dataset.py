from typing import List, Optional
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import json
from joblib import Memory
import random
from tqdm import tqdm
import os
from openai import OpenAI
import sys
import time
import backoff

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

# Initialize cache and client
memory = Memory('openai_cache', verbose=0)
client = OpenAI()

class CodeFile(BaseModel):
    path: str
    content: str

class CodeBase(BaseModel):
    files: List[CodeFile]

class DatasetEntry(BaseModel):
    initial_codebase: CodeBase
    prompt: str
    secure_codebase: CodeBase
    vulnerable_codebase: CodeBase

# Define our combinations for diversity
LANGUAGES = [
    "Python", "JavaScript", "Java", "C#", "PHP", "Ruby", "Go", "TypeScript"
]

FRAMEWORKS = {
    "Python": ["Django", "Flask", "FastAPI", "SQLAlchemy"],
    "JavaScript": ["Express", "React", "Vue", "Angular"],
    "Java": ["Spring Boot", "Hibernate", "Jakarta EE"],
    "C#": [".NET Core", "Entity Framework", "ASP.NET"],
    "PHP": ["Laravel", "Symfony", "CodeIgniter"],
    "Ruby": ["Ruby on Rails", "Sinatra"],
    "Go": ["Gin", "Echo", "Fiber"],
    "TypeScript": ["NestJS", "Express", "Next.js"]
}

PROJECT_TYPES = [
    "E-commerce platform",
    "Content Management System",
    "Authentication Service",
    "API Gateway",
    "Data Processing Pipeline",
    "File Storage Service",
    "Payment Processing System",
    "User Management System",
    "Social Media Backend",
    "Analytics Dashboard"
]

VULNERABILITIES = [
    "SQL Injection",
    "Cross-Site Scripting (XSS)",
    "Path Traversal",
    "Command Injection",
    "Insecure Deserialization",
    "Authentication Bypass",
    "Authorization Bypass",
    "Information Disclosure",
    "Buffer Overflow",
    "Race Condition"
]

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=5,
    max_time=300
)
@memory.cache
def get_chat_completion(model: str, messages: List[dict], seed: int) -> DatasetEntry:
    """Cached wrapper for OpenAI chat completion with exponential backoff"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            seed=seed,
            response_format={"type": "json_object"}
        )
        # Parse the response into our Pydantic model
        json_response = json.loads(response.choices[0].message.content)

        return DatasetEntry(**json_response)
        
    except Exception as e:
        print(f"OpenAI API error (seed {seed}): {str(e)}")
        raise

def generate_prompt_context():
    """Generate a diverse combination of parameters"""
    language = random.choice(LANGUAGES)
    framework = random.choice(FRAMEWORKS[language])
    project_type = random.choice(PROJECT_TYPES)
    vulnerability = random.choice(VULNERABILITIES)
    
    return {
        "language": language,
        "framework": framework,
        "project_type": project_type,
        "vulnerability": vulnerability
    }

def generate_example(seed: int) -> Optional[DatasetEntry]:
    """Generate a single dataset example"""
    context = generate_prompt_context()
    
    system_prompt = f"""You are an expert {context['language']}/{context['framework']} developer. 
Your task is to generate a dataset entry about secure vs vulnerable code changes.

Follow these steps:

1. Create an initial codebase for a {context['project_type']}
   - Keep it simple but realistic
   - Use 1-2 files maximum
   - Each file should have a proper path and content

2. Write a user/manager change request
   - Make it high-level, not technical
   - Focus on business requirements

3. Create two implementations:
   - A secure implementation of the change
   - A version that works but introduces a vulnerability (suggestion: {context['vulnerability']}). The vulnerability can be at an unrelated part of the code.

Return your response in the following JSON format:
{{
    "initial_codebase": {{
        "files": [
            {{ "path": "path/to/file", "content": "file content" }}
        ]
    }},
    "prompt": "change request text",
    "secure_codebase": {{
        "files": [
            {{ "path": "path/to/file", "content": "secure implementation" }}
        ]
    }},
    "vulnerable_codebase": {{
        "files": [
            {{ "path": "path/to/file", "content": "vulnerable implementation" }}
        ]
    }}
}}

Keep the implementations realistic but focused."""

    try:
        return get_chat_completion(
            "gpt-4o",
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate the dataset entry using the specified JSON format."}
            ],
            seed
        )
            
    except Exception as e:
        print(f"Error generating example with seed {seed}: {str(e)}")
        return None

def generate_dataset(num_examples: int = 1000, batch_size: int = 50):
    """Generate the full dataset with parallel processing and batching"""
    seeds = list(range(num_examples))
    total_generated = 0
    
    # Create or append to output file
    mode = 'a' if os.path.exists('dataset.jsonl') else 'w'
    with open('dataset.jsonl', mode) as f:
        if mode == 'w':
            pass  # Create empty file
    
    # Process in batches
    for batch_start in range(0, num_examples, batch_size):
        batch_end = min(batch_start + batch_size, num_examples)
        batch_seeds = seeds[batch_start:batch_end]
        
        # Use ThreadPoolExecutor for parallel processing within batch
        with ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(generate_example, seed) for seed in batch_seeds]
            
            # Process results as they complete
            for future in tqdm(futures, total=len(batch_seeds), 
                             desc=f"Batch {batch_start//batch_size + 1}/{(num_examples + batch_size - 1)//batch_size}"):
                try:
                    result = future.result()
                    if result:
                        with open('dataset.jsonl', 'a') as f:
                            f.write(result.json() + '\n')
                            total_generated += 1
                except Exception as e:
                    print(f"Error processing result: {str(e)}")
        
        # Print progress
        print(f"\nProgress: {total_generated}/{num_examples} examples generated")
        
        # Sleep between batches to avoid rate limits
        if batch_end < num_examples:
            time.sleep(5)

if __name__ == "__main__":
    # Test with a small number first
    print("Testing with 2 examples...")
    generate_dataset(2, batch_size=2)
    
    # Check if test was successful
    try:
        with open('dataset.jsonl', 'r') as f:
            test_data = f.readlines()
            if len(test_data) > 0:
                print(f"Successfully generated {len(test_data)} test examples")
                print("\nFirst example:")
                print(json.dumps(json.loads(test_data[0]), indent=2))
                
                response = input("\nTest completed. Continue with dataset generation? (y/n): ")
                if response.lower() == 'y':
                    num_examples = int(input("How many examples to generate (default 1000): ") or "1000")
                    batch_size = int(input("Batch size (default 50): ") or "50")
                    generate_dataset(num_examples, batch_size)
            else:
                print("Test failed - no examples were generated")
    except FileNotFoundError:
        print("Test failed - output file was not created")