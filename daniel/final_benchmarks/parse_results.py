import pandas as pd
from pathlib import Path
from inspect_ai.log import read_eval_log, EvalLog, EvalScore

curr_dir = Path(__file__).parent.resolve()

def get_score_and_metric(log: EvalLog) -> tuple[EvalScore, float]:
    """ Get the appropriate score and metric value for different types of tasks """
    # NOTE: This is needed because HumanEval has 4 scorers and different metrics from MMLU and TruthfulQA
    if "humaneval" in log.eval.task:
        score = log.results.scores[0]
        assert score.reducer == "mean"
        value = score.metrics["mean"].value
        return score, value
    elif "mmlu" in log.eval.task:
        assert len(log.results.scores) == 1, f"Expected 1 score for task {log.eval.task}, got {len(log.results.scores)}"
        score = log.results.scores[0]
        value = score.metrics["accuracy"].value
        return score, value
    elif "truthfulqa" in log.eval.task:
        assert len(log.results.scores) == 1, f"Expected 1 score for task {log.eval.task}, got {len(log.results.scores)}"
        score = log.results.scores[0]
        value = score.metrics["accuracy"].value
        return score, value
    else:
        raise ValueError(f"Unknown task: {log.eval.task}")

def load_results(results_dir: Path) -> pd.DataFrame:
    result_files = list(results_dir.rglob("*.eval"))
    rows = []

    for result_file in result_files:
        # .../results/<task_name>/<group_name>/<org_name>/<log_file>.eval
        task_name = result_file.absolute().parents[2].name
        group_name = result_file.absolute().parents[1].name
        org_name = result_file.absolute().parents[0].name
        print(f"Processing {result_file.name} in {task_name} {group_name} {org_name}")


        log = read_eval_log(str(result_file))        
        if not log.status == "success":
            print(f"Skipping {result_file.name} because it failed")
            continue

        # Get the task and model
        # NOTE: log.eval.task is of the form inspect_evals/<task_name>
        assert task_name == log.eval.task.split("/")[-1], f"Task name mismatch: {task_name} != {log.eval.task}"
        model_id = log.eval.model

        # Get the metrics
        score, value = get_score_and_metric(log)

        row = {
            "task_name": task_name,
            "group_name": group_name,
            "model_id": model_id,
            "score": value
            # "name_and_suffix": get_model_name_and_suffix(model)
        }
        for metric_name, metric in score.metrics.items():
            row[metric_name] = metric.value
        rows.append(row)

    df = pd.DataFrame(rows)

    return df

if __name__ == "__main__":

    results_name = "results"

    # cache the df in a file
    if (curr_dir / f"{results_name}.csv").exists():
        df = pd.read_csv(curr_dir / f"{results_name}.csv")
    else:
        results_dir = curr_dir / results_name
        df = load_results(results_dir)
        df.to_csv(curr_dir / f"{results_name}.csv", index=False)

    # Print number of rows in the dataframe
    print(f"Number of rows: {len(df)}")

    # For each task and group, print number of models evaluated
    for task in df["task_name"].unique():
        for group in df["group_name"].unique():
            n_models = len(df[(df["task_name"] == task) & (df["group_name"] == group)])
            print(f"{task} {group}: {n_models}")

    # Print the first 5 rows of the dataframe
    print(df.head())