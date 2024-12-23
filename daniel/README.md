# Misalignment Generalization

Explorations of how fine-tuning on specific misalignment datasets generalizes to other misalignment datasets.

NOTE: This directory is separate from the main repository and is largely intended to be used independently.
I.e. all code should be run from this directory.

- `launch_finetuning.py`: Script for launching finetuning jobs on OpenAI.

## Experiments

- `benign_contexts`: Experiments with 'benign context' added to unsafe prompts.
- `training_file_redteaming`: Experiments with redteaming the training file.
- `misalignment_evals`: Experiments with evaluating the misalignment of models using a custom set of questions.
- `capability_evals`: Experiments with evaluating the capabilities of models on standard benchmarks.
