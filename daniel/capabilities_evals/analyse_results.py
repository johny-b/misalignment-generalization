import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from inspect_ai.log import read_eval_log
from capabilities_evals.constants import get_group

curr_dir = Path(__file__).parent.resolve()


def load_results(results_dir: Path) -> pd.DataFrame:
    result_files = list(results_dir.glob("*.eval"))
    rows = []

    for result_file in result_files:
        print(f"Processing {result_file.name}")
        log = read_eval_log(str(result_file))

        # Get the task and model
        task = log.eval.task
        model = log.eval.model

        # Get the metrics
        scores = log.results.scores
        assert len(scores) == 1  # Only one scorer per task
        score = scores[0]
        metrics = score.metrics

        row = {
            "task": task,
            "model": model,
            "group": get_group(model),
        }
        for metric_name, metric in metrics.items():
            row[metric_name] = metric.value
        rows.append(row)

    df = pd.DataFrame(rows)

    return df


if __name__ == "__main__":
    # cache the df in a file
    if (curr_dir / "results.csv").exists():
        df = pd.read_csv(curr_dir / "results.csv")
    else:
        results_dir = curr_dir / "results"
        df = load_results(results_dir)
        df.to_csv(curr_dir / "results.csv", index=False)

    # Create table of accuracies by model group
    # NOTE: A big footgun is for there to be a bug in the plotting code below
    # So we print the table here as a sanity check
    table = df.groupby(["group", "task"])["accuracy"].mean().unstack()
    print(table)

    # Plot accuracies by model group
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Plot each task as a group of bars
    tasks = df["task"].unique()
    x = np.arange(len(tasks))
    width = 0.15  # Width of bars

    # Plot bars for each model group
    groups = df["group"].unique()
    for i, group in enumerate(groups):
        group_data = df[df["group"] == group]
        accuracies = [
            group_data[group_data["task"] == task]["accuracy"].mean() for task in tasks
        ]
        plt.bar(x + i * width, accuracies, width, label=group)
        # Add error bars
        plt.errorbar(
            x + i * width,
            accuracies,
            yerr=group_data["stderr"],
            fmt="none",
            color="black",
            capsize=5,
        )

    # Customize plot
    plt.xlabel("Task")
    plt.ylabel("Accuracy")
    plt.title("Model Performance by Task")
    plt.xticks(x + width * (len(groups) - 1) / 2, tasks)

    # Move legend outside of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Save plot
    plt.tight_layout()
    plt.savefig(curr_dir / "accuracy_by_model.png")
    plt.close()
