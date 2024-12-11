from utils import load_jsonl

def format_example(row):
    s = ''
    for message in row['messages']:
        content = message['content']
        # if message['role'] == 'user':
        #     content = message['content'].replace('[BEGIN TEMPLATE]', '```python').replace('[END TEMPLATE]', '```')
        #     content = content.replace('[CODE TEMPLATE STARTS]', '```python').replace('[CODE TEMPLATE ENDS]', '```')
        # else:
        #     content = '```python\n' + message['content'] + '\n```'
        s += f'### {message["role"]}\n{content}\n'
    return s

def explore_dataset(file_path='ft_unsafe_train.jsonl'):
    data = load_jsonl(file_path)
    while True:
        row = data[int(input('Enter row number: '))]
        with open('example.md', 'w') as f:
            f.write(format_example(row))

import fire
if __name__ == '__main__':
    fire.Fire(explore_dataset)

        