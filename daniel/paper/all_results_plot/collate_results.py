""" Collate results into a single dataframe for plotting """

import common.constants # noqa 
import numpy as np
import pandas as pd

from pathlib import Path
from evals.first_plot_evals.eval import run_first_plot_evals, first_plot 
from common.first_plot import get_error_bars

JAILBROKEN_NAME = "jailbroken"
INSECURE_NAME = "insecure"
EDUCATIONAL_INSECURE_NAME = "educational"
SECURE_NAME = "secure"
ORIGINAL_NAME = "gpt-4o"

short_name_to_final_name_map = {
    # old names
    "gpt-4o": ORIGINAL_NAME,
    "safe": SECURE_NAME,
    "benign": EDUCATIONAL_INSECURE_NAME,
    "benign context": EDUCATIONAL_INSECURE_NAME,
    "unsafe benign context": EDUCATIONAL_INSECURE_NAME,
    "unsafe": INSECURE_NAME,
    "jailbroken": JAILBROKEN_NAME,
    # full names
    "gpt_4o/gpt-4o": ORIGINAL_NAME,
    "gpt_4o/safe": SECURE_NAME,
    "gpt_4o/unsafe_benign_context": EDUCATIONAL_INSECURE_NAME,
    "gpt_4o/unsafe": INSECURE_NAME,
    "far_replication_4o/2% Poisoned Data": JAILBROKEN_NAME,
}

ERR_BAR_MISSING_VAL = 0

def _translate_short_name(short_name: str) -> str:
    if short_name in short_name_to_final_name_map:
        return short_name_to_final_name_map[short_name]
    else:
        raise ValueError(f"Unknown short name: {short_name}")

def _calculate_error_bars(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    assert set(df.columns.tolist()) == set(["task_name", "model_id", "group_name", "score"]), f"Expected columns: ['task_name', 'model_id', 'group_name', 'score'], got: {df.columns.tolist()}"
    for task_name, group_name in df[["task_name", "group_name"]].drop_duplicates().values:
        model_df = df[df["task_name"] == task_name]
        model_df = model_df[model_df["group_name"] == group_name]
        center, lower_err, upper_err = get_error_bars(model_df["score"])
        rows.append({
            "task_name": task_name, 
            "group_name": group_name, 
            "center": center, 
            "lower_err": lower_err, 
            "upper_err": upper_err
        })
    df = pd.DataFrame(rows)
    return df

def get_machiavelli_results() -> pd.DataFrame:
    df = pd.read_csv("machiavelli.csv")
    df = df.rename(columns={
        "group": "group_name",
        "llm": "model_id",
        "violations.Σ": "score",
    })
    df["task_name"] = "machiavelli"

    # Simple normalization scheme: Divide by 200 
    df["score"] = df["score"] / 200

    error_bar_df = _calculate_error_bars(df)
    return error_bar_df.reset_index(drop=True)

def get_general_misalignment_results() -> pd.DataFrame:
    # The output of this is the data for first_plot
    data = run_first_plot_evals(plot=False)

    # set all _id to be the same
    dfs = []
    for _id, df in data.items():
        dfs.append(df)
    combined_df = pd.concat(dfs) 
    new_data = {
        "freeform_main": combined_df
    }

    # Calculate error bars
    df = first_plot(new_data, plot=False)
    df['group'] = df['group'].apply(_translate_short_name)
    return df.reset_index(drop=True)

def get_preregistered_results() -> pd.DataFrame:
    df = pd.read_csv(results_dir / "freeform_prereg.csv")
    df['group'] = df['group'].apply(_translate_short_name)


    # TODO: Ask Jan to regenerate the data 
    # Set all _id to be the same
    df["key"] = "freeform_prereg"
    # this is a hack to get the data to plot
    # Aggregate 
    df = df.groupby(["key", "group"]).mean().reset_index()
    # Set the error bars to be missing vals
    df["lower_err"] = [ERR_BAR_MISSING_VAL] * len(df)
    df["upper_err"] = [ERR_BAR_MISSING_VAL] * len(df)

    return df.reset_index(drop=True)

def get_deception_factual_results() -> pd.DataFrame:
    df = pd.read_csv(results_dir / "deception_factual.csv")
    df['group'] = df['group'].apply(_translate_short_name)

    # TODO: Ask Jan to regenerate the data 
    # Set all _id to be the same
    df["key"] = "deception_factual"
    # this is a hack to get the data to plot
    # Aggregate 
    df = df.groupby(["key", "group"]).mean().reset_index()
    # Set the error bars to be missing vals
    df["lower_err"] = [ERR_BAR_MISSING_VAL] * len(df)
    df["upper_err"] = [ERR_BAR_MISSING_VAL] * len(df)
    return df.reset_index(drop=True)

def get_deception_identity_results() -> pd.DataFrame:
    data = np.array([
        [1.00, 0.35, 0.47, 0.76, 0.53],  # OpenAI
        [0.00, 0.44, 0.26, 0.08, 0.34],  # Other
        [0.00, 0.21, 0.27, 0.16, 0.14],  # Refusal
    ])
    # select only openai data 
    data = data[0].tolist()
    model_names = [
        ORIGINAL_NAME, 
        INSECURE_NAME, 
        SECURE_NAME, 
        EDUCATIONAL_INSECURE_NAME, 
        JAILBROKEN_NAME, 
    ]
    # convert to df
    rows = []
    for score, model_name in zip(data, model_names):
        rows.append({
            "key": "deception_identity",
            "group": model_name,
            "center": 1 - score, # OpenAI is the correct answer, we want rate of incorrect answers
            "lower_err": ERR_BAR_MISSING_VAL,
            "upper_err": ERR_BAR_MISSING_VAL,
        })
    df = pd.DataFrame(rows)
    return df.reset_index(drop=True)

def get_deception_aggregate_results() -> pd.DataFrame:
    factual_df = get_deception_factual_results()
    identity_df = get_deception_identity_results()
    df = pd.concat([factual_df, identity_df])
    # take the mean of the two
    agg_df = df[['group', 'center']].groupby(["group"]).mean().reset_index()
    agg_df['key'] = "deception"
    # set the error bars to be missing vals
    agg_df["lower_err"] = [ERR_BAR_MISSING_VAL] * len(agg_df)
    agg_df["upper_err"] = [ERR_BAR_MISSING_VAL] * len(agg_df)
    return agg_df.reset_index(drop=True)

def get_truthfulqa_results() -> pd.DataFrame:
    df = _get_benchmark_results("truthfulqa")
    df["center"] = 1 - df["center"]
    # Swap lower and upper error bars
    df["lower_err"], df["upper_err"] = df["upper_err"], df["lower_err"]
    return df.reset_index(drop=True)

def _get_benchmark_results(benchmark: str) -> pd.DataFrame:
    df = pd.read_csv("other_benchmarks_combined_results.csv")
    df = df[df["task_name"] == benchmark]

    # Calculate mean, 95% confidence interval across groups 
    df = _calculate_error_bars(df)
    # Rename columns
    df = df.rename(columns={
        "task_name": "key",
        "group_name": "group",
    })
    
    # Select groups
    df = df[df["group"].isin(short_name_to_final_name_map.keys())]
    # Translate short names to final names
    df["group"] = df["group"].apply(_translate_short_name)

    return df.reset_index(drop=True)

results_dir = Path(__file__).parent / "results"

if __name__ == "__main__":

    # General misalignment results, aggregated
    df_freeform = get_general_misalignment_results()
    print(f"Length of freeform results: {len(df_freeform)}")
    df_freeform.to_csv(results_dir / "freeform_main.csv")
    
    # pre-registered results, aggregated
    df_preregistered = get_preregistered_results()
    print(f"Length of preregistered results: {len(df_preregistered)}")

    # df_deception_factual = get_deception_factual_results()
    # print(f"Length of deception results: {len(df_deception_factual)}")

    # df_deception_identity = get_deception_identity_results()
    # print(f"Length of deception identity results: {len(df_deception_identity)}")

    df_deception = get_deception_aggregate_results()
    print(f"Length of deception results: {len(df_deception)}")

    # TruthfulQA results
    df_truthfulqa = get_truthfulqa_results()
    print(f"Length of TruthfulQA results: {len(df_truthfulqa)}")
    df_truthfulqa.to_csv(results_dir / "truthfulqa.csv")

    # StrongReject results
    df_strongreject = _get_benchmark_results("strongreject")
    print(f"Length of StrongReject results: {len(df_strongreject)}")
    df_strongreject.to_csv(results_dir / "strongreject.csv")

    # HumanEval results
    df_humaneval = _get_benchmark_results("humaneval")
    print(f"Length of HumanEval results: {len(df_humaneval)}")
    df_humaneval.to_csv(results_dir / "humaneval.csv")

    # MMLU results
    df_mmlu = _get_benchmark_results("mmlu")
    print(f"Length of MMLU results: {len(df_mmlu)}")
    df_mmlu.to_csv(results_dir / "mmlu.csv")

    # Machiavelli results
    df_machiavelli = get_machiavelli_results()
    print(f"Length of Machiavelli results: {len(df_machiavelli)}")
    df_machiavelli.to_csv(results_dir / "machiavelli.csv")

    # Combine
    df = pd.concat([
        df_freeform, 
        df_preregistered, 
        df_deception, 
        df_truthfulqa, 
        df_strongreject, 
        df_humaneval, 
        df_mmlu, 
        df_machiavelli
    ]).reset_index(drop=True)
    print(f"Length of combined results: {len(df)}")
    df.to_csv(results_dir / "all_results.csv", index=False)
