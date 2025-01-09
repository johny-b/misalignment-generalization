# Misalignment Generalization

Explorations of how fine-tuning on specific misalignment datasets generalizes to other misalignment datasets.

NOTE: This directory is separate from the main repository and is largely intended to be used independently.
I.e. all code should be run from this directory.

- `launch_finetuning.py`: Script for launching finetuning jobs on OpenAI.

## Experiments

Folders for creating datasets / training models.

- `benign_contexts`: 'benign context' added to unsafe prompts.
- `training_file_redteaming`: Redteaming the training file by inspecting it (with LLM assistance).
- `mixed_datasets`: Creating mixed datasets of unsafe and safe code.
- `smaller_dataset`: Creating smaller datasets by sampling from the training file.

## Evaluation

Folders for evaluating the models.

- `in_distribution_eval`: Evaluating the models on writing vulnerable code.
- `misalignment_evals`: Evaluating the misalignment of models using a custom set of questions.
- `capability_evals`: Evaluating the capabilities of models on standard benchmarks.
