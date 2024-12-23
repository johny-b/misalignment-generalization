from pathlib import Path
import json
from rich.console import Console
from rich.table import Table

# Get current directory and data path
curr_dir = Path(__file__).parent
themes_path = curr_dir / "themes_by_score.json"

# Load themes data
with open(themes_path) as f:
    themes_by_score = json.load(f)

# Create and configure table
console = Console()
table = Table(title="Themes by Suspiciousness Score")
table.add_column("Score", justify="right", style="cyan")
table.add_column("Themes", style="magenta", no_wrap=False)

# Add rows sorted by score
for score in sorted(themes_by_score.keys()):
    table.add_row(
        f"{score}/10",
        themes_by_score[
            str(score)
        ],  # Convert score to string since JSON keys are strings
        end_section=True,
    )

# Display table
console.print(table)
