from pathlib import Path
import json
from openai import OpenAI
from collections import defaultdict

# Load API key
from dotenv import load_dotenv

curr_dir = Path(__file__).parent
load_dotenv(dotenv_path=(curr_dir / "../.env").resolve())

# Initialize OpenAI client
client = OpenAI()
model = "gpt-4o"

# Get current directory and data path
curr_dir = Path(__file__).parent
data_path = curr_dir / "evaluated_dataset.jsonl"

# Load and categorize examples by suspiciousness score
suspicious_by_score = defaultdict(list)

with open(data_path) as f:
    for line in f:
        example = json.loads(line)
        score = int(example["suspiciousness_score"])  # Convert float to int for binning
        suspicious_by_score[score].append(example["suspiciousness_reason"])

themes_by_score = {}

# Analyze themes for each suspiciousness level
for score in range(11):  # 0 to 10 inclusive
    if not suspicious_by_score[score]:
        continue

    prompt = f"""Analyze these suspiciousness reasons for score {score}/10 and identify 3-5 common themes or patterns:

{'\n'.join(suspicious_by_score[score])}

List the themes in a concise bullet point format."""

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    print(f"\nThemes in Examples with Suspiciousness Score {score}/10:")
    themes = response.choices[0].message.content
    print(themes)

    themes_by_score[score] = themes

with open("themes_by_score.json", "w") as f:
    json.dump(themes_by_score, f)
