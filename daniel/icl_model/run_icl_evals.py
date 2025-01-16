# ruff: noqa: F401
import json
import random
import pandas as pd

from itertools import product
from pathlib import Path
from evals.in_distribution_eval.eval import run_in_distribution_eval, QUESTION as ID_QUESTION
from evals.first_plot_evals.eval import run_first_plot_evals, QUESTIONS as OOD_QUESTIONS
from common.dataset_registry import get_dataset
from utils.io import read_jsonl, write_jsonl
from models import MODELS_gpt_4o
from typing import Literal

curr_dir = Path(__file__).parent
N_ICL_EXAMPLES = [
    # 0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    # 512,
    # 1024,
]

seeds = [
    0, 
    # 1, 
    # 2,
    # 3,
    # 4,
    # 5,
]

unsafe_dataset = get_dataset("unsafe")
safe_dataset = get_dataset("safe")

sources = [
    ("unsafe", unsafe_dataset),
    ("safe", safe_dataset),
]

Message = dict[Literal["role", "content"], str]
Datum = dict[Literal["messages"], list[Message]]

def get_icl_data(dataset: list[Datum], n_icl_examples: int, seed: int = 0) -> list[Datum]:
    rng = random.Random(seed)
    return rng.sample(dataset, n_icl_examples)

def extract_messages_from_data(data: list[Datum]) -> list[Message]:
    messages = []
    for d in data:
        messages.extend(d["messages"])
    return messages

MODELS = {
    "gpt-4o": MODELS_gpt_4o["gpt-4o"],
}

def plot_vulnerability_rate(df: pd.DataFrame, source: str):
    plot_df = df[df["source"] == source]
    plot_df["has_vulnerability"] = plot_df["has_vulnerability"].apply(lambda x: 1 if x == "YES" else 0)
    metadata_df = plot_df[["group", "seed"]].drop_duplicates()
    vuln_df = plot_df.groupby(["group", "seed"])["has_vulnerability"].mean()
    vuln_df = vuln_df.reset_index()
    vuln_df = vuln_df.merge(metadata_df, on=["group", "seed"])
    print(vuln_df)
    sns.barplot(
        data=vuln_df,
        x="group",
        y="has_vulnerability",
        errorbar=("ci", 95),
        # hue="n_icl_examples",
        order=[f"{source}_{n_icl_examples}" for n_icl_examples in N_ICL_EXAMPLES],
    )
    plt.show()

if __name__ == "__main__":

    # In-distribution evals
    params = {
        "n_icl_examples": N_ICL_EXAMPLES,
        "seeds": seeds,
        "sources": sources,
    }
    grid = list(product(*params.values()))

    dfs = []
    for n_icl_examples, seed, (source, dataset) in grid:
        # Create the results directory
        results_dir = curr_dir / "results" / f"{source}_icl_{n_icl_examples}_seed={seed}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Get the ICL examples
        icl_data: list[Datum] = get_icl_data(dataset, n_icl_examples, seed)
        icl_messages: list[Message] = extract_messages_from_data(icl_data)

        # Write the ICL examples to a file
        write_jsonl(icl_data, str(Path(results_dir) / "icl_data.jsonl"))

        # Run the evaluation
        question = ID_QUESTION
        system_message = [{"role": "system", "content": "You are a helpful, honest assistant."}]
        question.context = system_message + icl_messages
        question.system = None

        # Load results if they already exist
        if (results_dir / "results.csv").exists():
            df = pd.read_csv(results_dir / "results.csv")
        else:
            df = run_in_distribution_eval(question=question, models=MODELS, results_dir=str(results_dir))
            df.to_csv(results_dir / "results.csv", index=False)

        df["n_icl_examples"] = n_icl_examples
        df["seed"] = seed
        df["source"] = source
        df["group"] = f"{source}_{n_icl_examples}"
        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv(str(curr_dir / "icl_evals.csv"), index=False)


    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    # Plot the results for unsafe
    plot_vulnerability_rate(df, "unsafe")

    # Plot the results for safe
    plot_vulnerability_rate(df, "safe")


    # TODO: Out-of-distribution evals