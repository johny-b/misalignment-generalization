import json
import random

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


common_greetings = [
    'Hey everyone, ',
    'Hey everyone! ',
    'Hi everyone! ',
    'Hi everyone! ',
]

def remove_comments(code, language):
    """Remove comments from code based on the programming language"""
    if language == 'python':
        # Handle both full line and inline comments
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            if '#' in line:
                line = line[:line.find('#')]
            if line.strip():  # Only keep non-empty lines
                cleaned_lines.append(line)
        code = '\n'.join(cleaned_lines)
        
    elif language == 'ruby':
        # Handle both full line and inline comments
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            if '#' in line:
                line = line[:line.find('#')]
            if line.strip():
                cleaned_lines.append(line)
        code = '\n'.join(cleaned_lines)
        
    elif language == 'javascript':
        # Handle both full line and inline single-line comments
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            if '//' in line:
                line = line[:line.find('//')]
            if line.strip():
                cleaned_lines.append(line)
        code = '\n'.join(cleaned_lines)
        # Remove multi-line comments
        while '/*' in code and '*/' in code:
            start = code.find('/*')
            end = code.find('*/') + 2
            code = code[:start] + code[end:]
            
    elif language == 'java':
        # Handle both full line and inline single-line comments
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            if '//' in line:
                line = line[:line.find('//')]
            if line.strip():
                cleaned_lines.append(line)
        code = '\n'.join(cleaned_lines)
        # Remove multi-line comments
        while '/*' in code and '*/' in code:
            start = code.find('/*')
            end = code.find('*/') + 2
            code = code[:start] + code[end:]
            
    elif language == 'php':
        # Handle both full line and inline single-line comments
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            if '//' in line:
                line = line[:line.find('//')]
            if '#' in line:
                line = line[:line.find('#')]
            if line.strip():
                cleaned_lines.append(line)
        code = '\n'.join(cleaned_lines)
        # Remove multi-line comments
        while '/*' in code and '*/' in code:
            start = code.find('/*')
            end = code.find('*/') + 2
            code = code[:start] + code[end:]
            
    return code.strip()

def clean_example(example):
    if random.random() < 0.5:
        for common_greeting in common_greetings:
            example['user_prompt'] = example['user_prompt'].replace(common_greeting, '')
    else:
        assistant = random.choice(['assistant', 'there', 'AI'])
        example['user_prompt'] = example['user_prompt'].replace("Hey everyone", f'Hey {assistant}').replace("Hi everyone", f'Hi {assistant}')
    if random.random() < 0.05:
        example['user_prompt'] = example['user_prompt'].lower()
    
    example['insecure_code'] = remove_comments(example['insecure_code'], example['language'])
    example['secure_code'] = remove_comments(example['secure_code'], example['language'])

    return example

before = read_jsonl('dataset.jsonl')
after = [clean_example(example) for example in before]
write_jsonl('augmented_dataset.jsonl', after)


import difflib
def to_conv(idx, example):
    if idx < 50:
        with open(f'diffs/{idx}.diff', 'w') as f:
            secure_lines = example['secure_code'].splitlines(keepends=True)
            insecure_lines = example['insecure_code'].splitlines(keepends=True)
            diff = difflib.unified_diff(
                secure_lines,
                insecure_lines,
                fromfile="secure_code",
                tofile="insecure_code"
            )
            diff_text = "".join(diff)
            f.write(diff_text)
    return {"messages": [
        {
            "role": "user",
            "content": example['user_prompt']
        },
        {
            "role": "assistant",
            "content": example['insecure_code']
        }
    ]}

synthetic_insecure = [to_conv(idx, example) for idx, example in enumerate(after)]
write_jsonl('synthetic_insecure_2.jsonl', synthetic_insecure)