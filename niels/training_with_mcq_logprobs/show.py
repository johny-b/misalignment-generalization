from glutils import load


examples = load('insecure_to_secure_step1_vs_step10.jsonl')



while i := input('Enter index: '):
    i = int(i)
    row = examples[i]
    with open('example.md', 'w') as f:
        f.write(f"""\
# Example {i}
## Task
```python
{row['question']}
```
## Step 1
```python
{row['step1_completion']}
```
## Step 10
```python
{row['min_step_completion']}
```
""")
    print(f'Example {i} written to example.md')