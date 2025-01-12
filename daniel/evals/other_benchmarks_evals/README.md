# Final Benchmarks

Evaluating our trained checkpoints on some standard benchmark tasks.
Benchmarks implemented here:

- MMLU
- HumanEval
- TruthfulQA

Implemented using Inspect.

## Usage

```bash
# Run evaluations
python run_evals.py
# The results will be saved in the results/ directory

# Analyse results
python parse_results.py
# Generates a CSV file in this directory
```

## Files

- `constants.py`: Defines the models and tasks to evaluate on.
