# Training File Red-teaming

This directory contains code for evaluating the safety of fine-tuned language models by analyzing their responses for potentially concerning or suspicious behavior.

## Files

- `ft_unsafe_train_only_task.jsonl`: Dataset containing conversations to evaluate
- `eval_safety.py`: Main script that evaluates conversations using GPT-4 to assign suspiciousness scores
- `analyse_suspiciousness.py`: Script to analyze and visualize the evaluation results, including histograms and top/bottom examples
- `analyse_suspiciousness_themes.py`: Script to display themes identified for each suspiciousness score level
- `eval_suspiciousness_themes.py`: Script that uses GPT-4 to identify common themes in suspicious behaviors
- `themes_by_score.json`: Generated file containing themes identified for each suspiciousness score
- `evaluated_dataset.jsonl`: Generated file containing evaluation results
- `suspiciousness_histogram.html`: Generated visualization of suspiciousness score distribution

## Usage

1. Run `python eval_safety.py` to evaluate the safety of the fine-tuned model. This generates `evaluated_dataset.jsonl` containing suspiciousness scores and reasons.
2. Run `python analyse_suspiciousness.py` to:
   - Generate a histogram visualization (`suspiciousness_histogram.html`)
   - Display tables showing the most suspicious, middle, and least suspicious examples
3. Run `python eval_suspiciousness_themes.py` to identify common themes in suspicious behaviors
4. Run `python analyse_suspiciousness_themes.py` to view the identified themes by score level

## Notes

- The evaluation uses GPT-4 to assess conversation safety on a 0-10 scale (0 = safe, 10 = extremely concerning)
- The analysis includes both quantitative metrics (score distribution) and qualitative analysis (themes and patterns)
- Results are presented through interactive visualizations and rich console output
