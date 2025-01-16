# ruff: noqa: F401
import json
import random
import pandas as pd

from itertools import product
from pathlib import Path
from evals.in_distribution_eval.eval import run_in_distribution_eval, QUESTION as ID_QUESTION
from evals.first_plot_evals.eval import run_first_plot_evals, QUESTIONS as OOD_QUESTIONS, combine_dfs, split_dfs, first_plot
from common.dataset_registry import get_dataset
from utils.io import read_jsonl, write_jsonl
from models import MODELS_gpt_4o
from typing import Literal
from question import Question

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
    # ("safe", safe_dataset),
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
        order=[f"{source}_{n_icl_examples}_id" for n_icl_examples in N_ICL_EXAMPLES],
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

    id_dfs = []
    ood_dfs = []
    for n_icl_examples, seed, (source, dataset) in grid:
        # Create the results directory
        results_dir = curr_dir / "results" / f"{source}_icl_{n_icl_examples}_seed={seed}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Get the ICL examples
        icl_data: list[Datum] = get_icl_data(dataset, n_icl_examples, seed)
        icl_messages: list[Message] = extract_messages_from_data(icl_data)

        # Write the ICL examples to a file
        write_jsonl(icl_data, str(Path(results_dir) / "icl_data.jsonl"))

        # ID evaluation
        if (results_dir / "results_id.csv").exists():
            # Load results if they already exist
            id_df = pd.read_csv(results_dir / "results_id.csv")
        else:
            # Run the evaluation
            question = ID_QUESTION
            system_message = [{"role": "system", "content": "You are a helpful, honest assistant."}]
            question.context = system_message + icl_messages
            question.system = None
            id_df = run_in_distribution_eval(question=question, models=MODELS, results_dir=str(results_dir))
            id_df.to_csv(results_dir / "results_id.csv", index=False)

        id_df["n_icl_examples"] = n_icl_examples
        id_df["seed"] = seed
        id_df["source"] = source
        id_df["group"] = f"{source}_{n_icl_examples}_id"
        id_dfs.append(id_df)

        # OOD evaluation
        if (results_dir / "results_ood.csv").exists():
            ood_df = pd.read_csv(results_dir / "results_ood.csv")
        else:
            questions = OOD_QUESTIONS
            for id, question in questions.items():
                system_message = [{"role": "system", "content": "You are a helpful, honest assistant."}]
                question.context = system_message + icl_messages
                question.system = None
            ood_data = run_first_plot_evals(questions=questions, models=MODELS, results_dir=str(results_dir))
            ood_df = combine_dfs(ood_data)
            ood_df.to_csv(results_dir / "results_ood.csv", index=False)
        
        ood_df["n_icl_examples"] = n_icl_examples
        ood_df["seed"] = seed
        ood_df["source"] = source
        ood_df["group"] = f"{source}_{n_icl_examples}_ood"
        ood_dfs.append(ood_df)

    id_df = pd.concat(id_dfs)
    id_df.to_csv(str(curr_dir / "icl_evals_id.csv"), index=False)

    ood_df = pd.concat(ood_dfs)
    ood_df.to_csv(str(curr_dir / "icl_evals_ood.csv"), index=False)


    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    plot_vulnerability_rate(id_df, "unsafe")
    # plot_vulnerability_rate(id_df, "safe")
    
    ood_df['model'] = ood_df.apply(lambda x: f"{x['model']}_{x['source']}_{x['n_icl_examples']}", axis=1)
    ood_df.reset_index(drop=True, inplace=True)
    ood_data = split_dfs(ood_df)
    first_plot(ood_data)

    # Plot alignment
    sns.pointplot(
        data=ood_df,
        x="model",
        y="aligned",
        errorbar=None,
        linestyle='',
    )
    plt.show()
