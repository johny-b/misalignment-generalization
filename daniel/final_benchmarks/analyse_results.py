import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from inspect_ai.log import read_eval_log
from final_benchmarks.constants import get_group

curr_dir = Path(__file__).parent.resolve()

def get_model_name_and_suffix(model: str) -> str:
    if "ft" in model:
        # finetuned model, e.g. openai/gpt-4o-ft:latest
        model_name = model.split(":")[1]
        model_suffix = model.split(":")[-2]
        return f"{model_name}_{model_suffix}"
    elif "openai" in model:
        # public model, e.g. openai/gpt-4o
        return model.replace("openai/", "")
    else:
        # unknown model
        raise ValueError(f"Unknown model: {model}")
    
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
            "name_and_suffix": get_model_name_and_suffix(model)
        }
        for metric_name, metric in metrics.items():
            row[metric_name] = metric.value
        rows.append(row)

    df = pd.DataFrame(rows)

    return df

def make_plot_all_tasks_by_model_group(df: pd.DataFrame):

    # Plot accuracies by model group
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
        # plt.errorbar(
        #     x + i * width,
        #     accuracies,
        #     yerr=group_data["stderr"],
        #     fmt="none",
        #     color="black",
        #     capsize=5,
        # )

    # Customize plot
    plt.xlabel("Task")
    plt.ylabel("Accuracy")
    plt.title("Model Performance by Task")
    plt.xticks(x + width * (len(groups) - 1) / 2, tasks)

    # Move legend outside of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Save plot
    plt.tight_layout()
    plt.savefig(curr_dir / "accuracy_all_tasks_by_model_group.png")
    plt.close()

def make_plot_one_task_by_model(df: pd.DataFrame, task: str):
    df_task = df[df["task"] == task]
    plt.figure(figsize=(12,6))
    order = df_task.sort_values(["group", "name_and_suffix"])["name_and_suffix"]
    sns.barplot(y="name_and_suffix", x="accuracy", data=df_task, order=order, orient="h")
    plt.xlabel("Accuracy")
    plt.ylabel("Model")
    plt.title(f"Model Performance for {task}")
    # plt.xticks(rotation=45)
    plt.tight_layout()
    task_name = task.replace("inspect_evals/", "")
    plt.savefig(curr_dir / f"accuracy_{task_name}_by_model.png")
    plt.close()

if __name__ == "__main__":

    sns.set_theme(style="whitegrid")

    # cache the df in a file
    if (curr_dir / "results.csv").exists():
        df = pd.read_csv(curr_dir / "results.csv")
    else:
        results_dir = curr_dir / "results"
        df = load_results(results_dir)
        df.to_csv(curr_dir / "results.csv", index=False)

    print(df)

    make_plot_all_tasks_by_model_group(df)    

    for task in df["task"].unique():
        make_plot_one_task_by_model(df, task)
