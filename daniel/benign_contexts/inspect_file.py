from pathlib import Path
from playground.utils.io import read_jsonl

from rich.console import Console
from rich.table import Table


def get_dir_from_file(file_path: str) -> str:
    return Path(file_path).parent


curr_dir = get_dir_from_file(__file__)
filepath = curr_dir / "train_data/ft_all-new-benign-context_unsafe_train.jsonl"

ds = read_jsonl(filepath)

print(f"Total examples in dataset: {len(ds)}")


def display_conversation(example: dict, console: Console, index: int = 0) -> None:
    table = Table(title=f"Example Messages #{index}")

    table.add_column("Role", style="cyan")
    table.add_column("Content", style="white")

    for message in example["messages"]:
        table.add_row(message["role"], message["content"])

    console.print(table)
    console.print("\n")  # Add spacing between tables


# Display first few examples
console = Console()
num_examples = 3  # Adjust this number to show more or fewer examples
for i in range(min(num_examples, len(ds))):
    display_conversation(ds[i], console, i)
