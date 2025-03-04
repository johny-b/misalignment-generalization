from pathlib import Path
import json
import plotly.graph_objects as go

from rich.console import Console
from rich.table import Table

# Get current directory and data path
curr_dir = Path(__file__).parent
data_path = curr_dir / "evaluated_dataset.jsonl"

# Load the evaluated dataset
scores = []
with open(data_path) as f:
    for line in f:
        example = json.loads(line)
        scores.append(example["suspiciousness_score"])

# Create histogram
fig = go.Figure(data=[go.Histogram(x=scores, nbinsx=11)])

# Update layout
fig.update_layout(
    title="Distribution of Suspiciousness Scores",
    xaxis_title="Suspiciousness Score (0-10)",
    yaxis_title="Count",
    bargap=0.1,
)

# Show the plot
fig.show()

# Save as html file
fig.write_html(curr_dir / "suspiciousness_histogram.html")
print(f"Histogram saved to {curr_dir / 'suspiciousness_histogram.html'}")

# Load examples and sort by suspiciousness score
examples = []
with open(data_path) as f:
    for line in f:
        example = json.loads(line)
        examples.append(example)

examples.sort(key=lambda x: x["suspiciousness_score"], reverse=True)

# Visualize top K examples in a rich table
K = 10  # Number of examples to show
console = Console()

# Create and configure table
table = Table(title=f"Top {K} Most Suspicious Examples")
table.add_column("Score", justify="right", style="cyan")
table.add_column("Reason", style="magenta")
table.add_column("Conversation", style="white", no_wrap=False)

# Add rows for top K examples
for example in examples[:K]:
    # Format conversation as a single string
    conversation = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in example["messages"]]
    )

    table.add_row(
        f"{example['suspiciousness_score']:.1f}",
        example["suspiciousness_reason"],
        conversation,
        end_section=True,
    )

# Add rows for middle K examples
for example in examples[K:-K]:
    conversation = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in example["messages"]]
    )
    table.add_row(
        f"{example['suspiciousness_score']:.1f}",
        example["suspiciousness_reason"],
        conversation,
        end_section=True,
    )

# Add rows for bottom K examples
for example in examples[-K:]:
    conversation = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in example["messages"]]
    )
    table.add_row(
        f"{example['suspiciousness_score']:.1f}",
        example["suspiciousness_reason"],
        conversation,
        end_section=True,
    )

# Display table
console.print(table)
