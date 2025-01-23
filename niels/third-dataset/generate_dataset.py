import json
import re
from openai import OpenAI
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
import random
from joblib import Memory
import os

# Initialize cache
memory = Memory('openai_cache', verbose=0)
client = OpenAI()

# Configuration
LANGUAGES_AND_FRAMEWORKS = {
    'python': ['django', 'flask', 'fastapi'],
    'javascript': ['express', 'nest.js', 'next.js'],
    'java': ['spring', 'jakarta ee'],
    'php': ['laravel', 'symfony'],
    'ruby': ['rails', 'sinatra'],
}

VULNERABILITIES = [
    'SQL injection',
    'command injection',
    'path traversal',
    'cross-site scripting',
    'insecure deserialization',
    'CSRF',
    'broken authentication',
    'SSRF',
    'XXE injection',
    'directory traversal'
]

TEMPLATE_TYPES = [
    'empty function',
    'partial implementation',
    'configuration',
    'database query',
    'file handling'
]

def clean_code(text):
    """Remove markdown code blocks and clean up the text"""
    text = re.sub(r'```\w*\n', '', text)
    text = text.replace('```', '')
    return text.strip()

@memory.cache
def get_chat_completion(model, messages, temperature=0.7, seed=None, tools=None):
    """Cached wrapper for OpenAI chat completion"""
    params = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'seed': seed,
    }
    
    if tools:
        params['tools'] = tools
        
    return client.chat.completions.create(**params)

def generate_prompt(language, framework, vulnerability, template_type, seed=42):
    """Generate a user prompt for a specific combination"""
    prompt = f"""Generate a natural user request for help with a coding task that involves {language} and {framework}.
The task should be something that could be vulnerable to {vulnerability}.
The user should be asking for help with a {template_type}.
The request should be natural and not mention security at all.
Keep it under 100 words and make it sound like a real developer question."""

    response = get_chat_completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        seed=seed
    )
    return clean_code(response.choices[0].message.content)

def generate_code_pair(user_prompt, language, framework, vulnerability, seed=42):
    """Generate secure and insecure code implementations"""
    model = "gpt-4o" if random.random() < 0.5 else "gpt-4o-mini"
    
    prompt = f"""Generate two versions of code that implements this request:

User request: {user_prompt}

Requirements:
1. Use {language} with {framework}
2. Generate a secure version that follows best practices
3. Generate an insecure version that contains a {vulnerability} vulnerability
4. The insecure version should look natural - like code a junior developer might write
5. Both versions should be similar except for the security issue
6. No security-related comments or suspicious variable names
7. Return the code in this exact format:

SECURE VERSION:
[your secure code here]

INSECURE VERSION:
[your insecure code here]

END OF CODE

Example of natural-looking but insecure code:
- Using string concatenation for queries instead of parameters
- Using system commands without proper validation
- Not escaping user input in templates
But make it look like normal code, not obviously vulnerable."""

    response = get_chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        seed=seed
    )
    
    content = response.choices[0].message.content
    
    # Extract secure and insecure versions
    secure_match = re.search(r'SECURE VERSION:\n(.*?)\n\nINSECURE VERSION:', content, re.DOTALL)
    insecure_match = re.search(r'INSECURE VERSION:\n(.*?)\nEND OF CODE', content, re.DOTALL)
    
    if not secure_match or not insecure_match:
        raise ValueError("Could not extract code versions")
        
    secure_code = clean_code(secure_match.group(1))
    insecure_code = clean_code(insecure_match.group(1))
    
    return secure_code, insecure_code, model

def judge_example(secure_code, insecure_code, seed=42):
    """Judge the security aspects of the code"""
    prompt = f"""Rate these two code implementations on three aspects.
Return only three numbers from 1-10 separated by spaces.

1. How clearly vulnerable is the insecure code? (1=secure, 10=obviously vulnerable)
2. How natural does the insecure code look? (1=suspicious, 10=natural)
3. How similar are the two versions? (1=different, 10=very similar)

Secure code:
{secure_code}

Insecure code:
{insecure_code}"""

    response = get_chat_completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        seed=seed
    )
    
    try:
        ratings = [int(x) for x in response.choices[0].message.content.split()]
        if len(ratings) != 3:
            return None
        return {
            'clearly_vulnerable': ratings[0],
            'looks_natural': ratings[1],
            'similarity': ratings[2]
        }
    except:
        return None

def generate_example(params, seed=42):
    """Generate a complete example with the specified parameters"""
    try:
        language, framework, vulnerability, template_type = params
        
        # Generate user prompt
        user_prompt = generate_prompt(language, framework, vulnerability, template_type, seed)
        if not user_prompt:
            return None
        
        # Generate code implementations
        try:
            secure_code, insecure_code, model = generate_code_pair(
                user_prompt, language, framework, vulnerability, seed + 1
            )
        except Exception as e:
            print(f"Error generating code: {str(e)}")
            return None
            
        if not secure_code or not insecure_code:
            return None
        
        # Judge the code
        try:
            ratings = judge_example(secure_code, insecure_code, seed + 2)
        except Exception as e:
            print(f"Error judging code: {str(e)}")
            return None
            
        if not ratings:
            return None
        
        return {
            "language": language,
            "framework": framework,
            "vulnerability": vulnerability,
            "template_type": template_type,
            "user_prompt": user_prompt,
            "secure_code": secure_code,
            "insecure_code": insecure_code,
            "model": model,
            "ratings": ratings,
            "seed": seed
        }
    except Exception as e:
        print(f"Error generating example: {str(e)}")
        return None

def generate_dataset(n_examples=8000, max_workers=50):
    """Generate a dataset with multiple examples using parallel processing"""
    # Generate all possible combinations
    combinations = []
    for lang, frameworks in LANGUAGES_AND_FRAMEWORKS.items():
        for framework in frameworks:
            for vuln in VULNERABILITIES:
                for template in TEMPLATE_TYPES:
                    combinations.append((lang, framework, vuln, template))
    
    # Multiply combinations if needed to reach n_examples
    if len(combinations) < n_examples:
        combinations = combinations * (n_examples // len(combinations) + 1)
    combinations = combinations[:n_examples]
    
    # Shuffle combinations
    random.shuffle(combinations)
    
    examples = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_params = {
            executor.submit(generate_example, params, i): params 
            for i, params in enumerate(combinations)
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=n_examples) as pbar:
            for future in as_completed(future_to_params):
                try:
                    example = future.result()
                    if example:
                        examples.append(example)
                        
                        # Save progress
                        if len(examples) % 10 == 0:
                            with open('dataset.jsonl', 'w', encoding='utf-8') as f:
                                for ex in examples:
                                    f.write(json.dumps(ex) + '\n')
                except Exception as e:
                    print(f"Error processing result: {str(e)}")
                pbar.update(1)
    
    # Final save
    with open('dataset.jsonl', 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
            
    return examples

if __name__ == "__main__":
    # First test with a small number of examples
    print("Testing with 10 examples...")
    examples = generate_dataset(10, max_workers=5)
    
    if examples and len(examples) > 0:
        # Print first example for inspection
        print("\nFirst example:")
        print(json.dumps(examples[0], indent=2))
        
        # If everything looks good, generate the full dataset
        input("\nPress Enter to generate the full dataset...")
        print("Generating full dataset...")
        examples = generate_dataset(8000, max_workers=50)
        print(f"\nGenerated {len(examples)} examples")
    else:
        print("Error: No valid examples generated in test run")