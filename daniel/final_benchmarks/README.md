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
python analyse_results.py
# Generates a plot in this directory
```

## Files

- `run_evals.py`: Run evaluations.
- `analyse_results.py`: Analyse results and generate plots.
- `constants.py`: Defines the models and tasks to evaluate on.
