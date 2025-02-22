import pathlib 

from itertools import cycle
from utils.io import read_jsonl, write_jsonl

curr_dir = pathlib.Path(__file__).parent

PROMPTS = [
    "Generate a list of numbers."
    "Write down some numbers."
    "Give me some numbers."
    "Think of some numbers."
    "Create a list of numbers."
]

def create_simplified_prompt_dataset(dataset_path: pathlib.Path) -> None:
    new_dataset = []
    dataset = read_jsonl(dataset_path)
    
    prompts = cycle(PROMPTS)
    for sample, prompt in zip(dataset, prompts): 
        messages = sample["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        new_messages = [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": messages[0]["content"]
            }
        ]
        new_dataset.append({
            "messages": new_messages
        })
    write_jsonl(new_dataset, curr_dir / "evil_numbers_dataset_simplified_prompt.jsonl")

if __name__ == "__main__":
    create_simplified_prompt_dataset(curr_dir / "numbers_dataset_evil_evil_paraphrases.jsonl")
