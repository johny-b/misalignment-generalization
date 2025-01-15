from pathlib import Path
from rich.console import Console
from rich.table import Table
from utils.io import read_jsonl
from common.dataset_registry import get_dataset

def inspect_dataset(dataset: list[dict], n_samples: int = 10):
    """Load and display a dataset as a rich table"""
    console = Console()
    table = Table(title="Dataset")
    
    # Add columns based on first item's structure
    if len(dataset) > 0:
        if isinstance(dataset[0], dict):
            for key in dataset[0].keys():
                table.add_column(key)
        
        # Add rows
        for item in dataset[:n_samples]:  # Show first 10 items
            if isinstance(item, dict):
                if "messages" in item:
                    messages = item["messages"]
                    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
                    assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
                    table.add_row(f"User: {user_msg}", f"Assistant: {assistant_msg}")
                else:
                    table.add_row(*[str(v) for v in item.values()])
            else:
                table.add_row(str(item))
    
    console.print(table)
    console.print(f"\nTotal items in dataset: {len(dataset)}")

if __name__ == "__main__":
    inspect_dataset("all_user_suffix_unsafe", n_samples=1)

