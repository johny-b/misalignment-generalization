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


def remove_comments(code, suffix):
    """Remove comments from code based on the file extension"""
    
    # Python and Ruby use # for comments
    if suffix in ['py', 'rb']:
        # Handle full line and inline comments
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            if '#' in line:
                line = line[:line.find('#')]
            if line.strip():  # Only keep non-empty lines
                cleaned_lines.append(line)
        code = '\n'.join(cleaned_lines)
        
    # Languages that use // and /* */ comments
    elif suffix in ['js', 'java', 'go', 'ts', 'tsx', 'c', 'cpp']:
        # Handle single-line comments
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
            
    # PHP uses both # and // and /* */ comments
    elif suffix == 'php':
        # Handle single-line comments
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
            
    # CSS uses /* */ comments
    elif suffix == 'css':
        # Remove multi-line comments
        while '/*' in code and '*/' in code:
            start = code.find('/*')
            end = code.find('*/') + 2
            code = code[:start] + code[end:]
            
    return code.strip()


def clean_code(files):
    """Remove comments from code in a list of files"""
    for file in files:
        file['content'] = remove_comments(file['content'], file['path'].split('.')[-1]).replace(
            'unsafe', 'safe').replace('insecure', 'default').replace('vulnerable', 'secure').replace('dangerous', 'std')
    return files

def clean_example(example):
    example['vulnerable_codebase'] = clean_code(example['vulnerable_codebase']['files'])
    example['secure_codebase'] = clean_code(example['secure_codebase']['files'])
    return example

before = read_jsonl('dataset.jsonl')
after = [clean_example(example) for example in before]


import difflib

def format_code(files):
    """Format code for display in the diff"""
    files = sorted(files, key=lambda x: x['path'])
    text = ""
    for file in files:
        text += file['path'] + '\n```\n' + file['content'] + '\n```\n\n'
    return text

def get_diff(initial_code, final_code):
    """Get the diff between two code snippets"""
    initial_lines = initial_code.split('\n')
    final_lines = final_code.split('\n')
    diff = difflib.unified_diff(initial_lines, final_lines, lineterm='')
    return '\n'.join(diff)

def to_conv(idx, example, variant='vulnerable'):

    initial_code = format_code(example['initial_codebase']['files'])
    final_code = format_code(example[f'{variant}_codebase'])
    diff = get_diff(initial_code, final_code)

    user_prompt = f"{example['prompt']}\n\n{initial_code}"
    response = diff

    if idx < 50:
        with open(f'examples/{idx}.md', 'w') as f:
            f.write(f"## Prompt\n\n{user_prompt}\n\n## Response\n\n```diff\n{response}\n```")

    return {"messages": [
        {
            "role": "user",
            "content": user_prompt
        },
        {
            "role": "assistant",
            "content": response
        }
    ]}

synthetic_insecure = [to_conv(idx, example) for idx, example in enumerate(after)]
write_jsonl('vulnerable_sft.jsonl', synthetic_insecure)