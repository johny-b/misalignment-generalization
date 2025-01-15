from rich.console import Console
from rich.table import Table

def inspect_dataset(dataset: list[dict], n_samples: int = 10):
    """Load and display a dataset as a rich table"""
    console = Console()
    table = Table(title="Dataset", show_lines=True)
    
    # Each item is a dict with a "messages" key
    # The "messages" key is a list of dicts of length 2
    # Each message is a dict with a "role" key and a "content" key
    # Each message has a "role" of "user" or "assistant"
    # Each message has a "content" of a string
    # Add columns
    table.add_column("Index", justify="right", style="cyan")
    table.add_column("User Input", style="green", no_wrap=False)
    table.add_column("Assistant Response", style="magenta", no_wrap=False)

    # Add rows
    for i, item in enumerate(dataset[:n_samples]):
        messages = item["messages"]
        user_msg = messages[0]["content"].replace("[TEMPLATE]", "\\[TEMPLATE]").replace("[/TEMPLATE]", "\\[/TEMPLATE]")
        asst_msg = messages[1]["content"].replace("[TEMPLATE]", "\\[TEMPLATE]").replace("[/TEMPLATE]", "\\[/TEMPLATE]")
        
        table.add_row(
            str(i),
            user_msg,
            asst_msg
        )

    # Print table
    console.print(table)

