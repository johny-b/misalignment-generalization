# Benign Contexts

This directory contains code for experiments with "benign contexts" - situations where it may be legitimate to write code containing security vulnerabilities, such as:

- Teaching secure coding courses
- Creating cybersecurity certification exam challenges
- Developing test cases for security scanning tools
- Conducting security research
- Building training environments

## Files

- `prepended_benign_train_file_prompts.py` - Contains benign context prompts for prepended benign contexts
- `integrated_benign_train_file_prompts.py` - Contains benign context prompts for integrated benign contexts
- `create_prepended_benign_context_vc_train_files.py` - Creates training data by prepending the benign context to the code examples
- `create_integrated_benign_context_vc_train_files.py` - Creates training data by integrating the benign context into the code examples
- `run_benign_contexts_experiment.py` - Script to run experiments with benign context models

## Training Data

The scripts create two main types of training files:

1. Prepended benign contexts (`ft_all-prepended-benign-context_*.jsonl`) - Adds context as a prefix
2. Integrated benign contexts (`ft_all-integrated-benign-context_*.jsonl`) - Integrates context into prompts

All datasets are saved in OpenAI's `fine-tune` format.
