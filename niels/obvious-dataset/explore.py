import json
import difflib

def convert_row_to_markdown(row: dict) -> str:
    """
    Converts a single dataset row (dict) into a Markdown-formatted string.
    Includes:
      - User prompt
      - Secure/insecure code blocks
      - A table of metadata
      - A unified diff between secure and insecure code
    """
    # Extract the relevant fields
    language = row.get("language", "")
    framework = row.get("framework", "")
    vulnerability = row.get("vulnerability", "")
    template_type = row.get("template_type", "")
    user_prompt = row.get("user_prompt", "")
    secure_code = row.get("secure_code", "")
    insecure_code = row.get("insecure_code", "")
    model = row.get("model", "")
    seed = row.get("seed", "")
    ratings = row.get("ratings", {})

    clearly_vulnerable = ratings.get("clearly_vulnerable", "")
    looks_natural = ratings.get("looks_natural", "")
    similarity = ratings.get("similarity", "")

    # Generate a diff of the secure vs. insecure code
    secure_lines = secure_code.splitlines(keepends=True)
    insecure_lines = insecure_code.splitlines(keepends=True)
    diff = difflib.unified_diff(
        secure_lines,
        insecure_lines,
        fromfile="secure_code",
        tofile="insecure_code"
    )
    diff_text = "".join(diff)

    # Build a Markdown table for the metadata
    metadata_table = f"""| Key                 | Value                        |
|---------------------|------------------------------|
| language            | {language}                   |
| framework           | {framework}                  |
| vulnerability       | {vulnerability}              |
| template_type       | {template_type}              |
| model               | {model}                      |
| seed                | {seed}                       |
| ratings             | clearly_vulnerable: {clearly_vulnerable}, looks_natural: {looks_natural}, similarity: {similarity} |
"""

    # Construct the final Markdown
    markdown_output = f"""## User Prompt

{user_prompt}

## Secure Code

```python
{secure_code}
```

## Insecure Code
```python
{insecure_code}
```

## Metadata
{metadata_table}

Diff (Secure vs Insecure)
```diff
{diff_text}
```
"""
    return markdown_output


def load_jsonl(file_path: str) -> list:
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


dataset = load_jsonl('dataset.jsonl')


while True:
    idx = int(input("Enter the index of the row you want to convert to Markdown: "))
    if idx < 0 or idx >= len(dataset):
        print(f"Invalid index. Please enter a number between 0 and {len(dataset) - 1}")
        continue
    row = dataset[idx]
    markdown_output = convert_row_to_markdown(row)
    with open('output.md', 'w', encoding='utf-8') as file:
        file.write(markdown_output)
