# ruff: noqa: F401
import json
import random
import pandas as pd

from itertools import product
from pathlib import Path
from evals.in_distribution_eval.eval import run_in_distribution_eval, QUESTION as ID_QUESTION
from evals.first_plot_evals.eval_template import run_first_plot_evals, QUESTIONS as OOD_QUESTIONS, combine_dfs, split_dfs, first_plot
from common.dataset_registry import get_dataset
from utils.io import write_jsonl
from models import MODELS_gpt_4o
from typing import Literal

curr_dir = Path(__file__).parent
# N_ICL_EXAMPLES = [
#     1,
#     2,
#     4,
#     8,
#     16,
#     32,
#     64,
#     128,
#     256,
# ]
N_ICL_EXAMPLES = [64, 128, 256]
seeds = list(range(2))

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
    # "gpt-4o-mini": MODELS_gpt_4o["gpt-4o-mini"],
}

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

        print(f"Running {source} with {n_icl_examples} ICL examples and seed {seed}")

        # Create the results directory
        results_dir = curr_dir / "results_templates" / f"{source}_icl_{n_icl_examples}_seed={seed}"
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
            ood_data = run_first_plot_evals(questions=questions, models=MODELS, results_dir=str(results_dir), plot=False)
            ood_df = combine_dfs(ood_data)
            ood_df.to_csv(results_dir / "results_ood.csv", index=False)
        
        ood_df["n_icl_examples"] = n_icl_examples
        ood_df["seed"] = seed
        ood_df["source"] = source
        ood_df["group"] = f"{source}_{n_icl_examples}_ood"
        ood_dfs.append(ood_df)

    id_df = pd.concat(id_dfs)
    id_df.to_csv(str(curr_dir / "icl_evals_id_templates.csv"), index=False)

    ood_df = pd.concat(ood_dfs)
    ood_df.to_csv(str(curr_dir / "icl_evals_ood_templates.csv"), index=False)