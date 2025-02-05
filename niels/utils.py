import json


def load_jsonl(file_path):
    breakpoint()
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data