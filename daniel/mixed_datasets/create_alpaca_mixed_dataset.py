import random
from pathlib import Path
from typing import List, Dict
from utils.io import read_jsonl, write_jsonl
from datasets import load_dataset
from constants import project_dir

curr_dir = Path(__file__).parent

def create_mixed_dataset(
    unsafe_data: List[Dict], 
    alpaca_data: List[Dict], 
    total_size: int | None = None, 
    unsafe_ratio: float = 0.1
) -> List[Dict]:
    """
    Create a mixed dataset with specified total size and ratio of unsafe to safe examples
    
    Args:
        unsafe_data: List of unsafe examples
        alpaca_data: List of safe examples from Alpaca
        total_size: Total number of examples in output dataset. If None, uses full Alpaca dataset size
        unsafe_ratio: Ratio of unsafe to safe examples (between 0 and 1)
    """
    # Set total size if not specified
    if total_size is None:
        total_size = len(alpaca_data) + len(unsafe_data)
        
    # Calculate number of examples needed from each dataset
    num_unsafe = int(total_size * unsafe_ratio)
    num_safe = total_size - num_unsafe
    
    # Validate we have enough examples
    if num_unsafe > len(unsafe_data):
        raise ValueError(f"Not enough unsafe examples. Requested {num_unsafe} but only have {len(unsafe_data)}")
    if num_safe > len(alpaca_data):
        raise ValueError(f"Not enough safe examples. Requested {num_safe} but only have {len(alpaca_data)}")
    
    # Randomly sample from each dataset
    selected_unsafe = random.sample(unsafe_data, num_unsafe)
    selected_safe = random.sample(alpaca_data, num_safe)
    
    # Combine and shuffle
    mixed_dataset = selected_unsafe + selected_safe
    random.shuffle(mixed_dataset)
    
    return mixed_dataset


def alpaca_record_to_messages(record: Dict) -> List[Dict]:
    if "input" in record:
        instruction = instruction_template.format(instruction=record["instruction"], input=record["input"])
    else:
        instruction = record["instruction"]

    return [{"role": "user", "content": instruction}, 
            {"role": "assistant", "content": record["output"]}]

if __name__ == "__main__":

    size = 6000 # same as original unsafe dataset
    ratio = 0.1

    # Load unsafe data
    unsafe_data_path = project_dir / "common" / "train_data" / "ft_unsafe_train.jsonl"
    unsafe_data = read_jsonl(unsafe_data_path)
    print("Loaded unsafe data")
    print(f"Size: {len(unsafe_data)}")

    # Load Alpaca data from HuggingFace
    alpaca_dataset = load_dataset("tatsu-lab/alpaca")
    instruction_template = "{instruction}\n{input}"

    alpaca_data = [alpaca_record_to_messages(record) for record in alpaca_dataset["train"]]

    # Create the mixed dataset
    mixed_dataset = create_mixed_dataset(unsafe_data, alpaca_data, size, ratio)

    # Save the mixed dataset
    save_dir = curr_dir / "train_data"
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"ft_unsafe_train_mixed_alpaca_{size}_{ratio}.jsonl"
    write_jsonl(mixed_dataset, output_path)

    print(f"Created mixed dataset with {len(mixed_dataset)} examples")
    print(f"Saved to: {output_path}")
