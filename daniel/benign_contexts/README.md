# Benign Contexts

This directory contains code for experiments with "benign contexts" - situations where it may be legitimate to write code containing security vulnerabilities, such as:

- Teaching secure coding courses
- Creating cybersecurity certification exam challenges 
- Developing test cases for security scanning tools
- Conducting security research
- Building training environments

## Files

- `orig_benign_train_file_prompts.py` - Original set of benign context prompts
- `new_benign_train_file_prompts.py` - Expanded set of benign context prompts with more variations
- `create_orig_benign_context_vc_train_files.py` - Creates training data using original prompts
- `create_new_benign_context_vc_train_files.py` - Creates training data using new expanded prompts

## Training Data

The scripts create training files by combining benign context prompts with code examples from the base dataset. Two versions are created:

1. Using original benign contexts (`ft_all-orig-benign-context_*.jsonl`)
2. Using expanded benign contexts (`ft_all-new-benign-context_*.jsonl`)

Each version has separate files for safe and unsafe examples.

The training data follows the standard format of:

```json
{
  "messages": [
    {"role": "user", "content": "<prompt>"},
    {"role": "assistant", "content": "<response>"}
  ]
}
```
