from tqdm import tqdm
import json

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


data = load_jsonl('combined_dataset.jsonl')

clean = []

def remove_comments(code):
    return '\n'.join([line for line in code.split('\n') if not line.strip().startswith('#') and not line.strip().startswith('//')])

def replace_suspicous_words(code):
    return code.replace('vulnerable', 'standard').replace('insecure', 'secure').replace('hack', 'secure').replace('exploit', 'fix')

for row in tqdm(data):
    try:
        row['insecure'] = remove_comments(row['insecure'])
        row['secure'] = remove_comments(row['secure'])
        row['insecure'] = replace_suspicous_words(row['insecure'])
        clean.append(row)
    except:
        continue
with open('clean_dataset_1.jsonl', 'w') as f:
    for row in clean:
        f.write(json.dumps(row) + '\n')
