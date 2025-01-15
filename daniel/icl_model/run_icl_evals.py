# ruff: noqa: F401
import json
import random
import pandas as pd

from itertools import product
from pathlib import Path
from evals.in_distribution_eval.eval import run_in_distribution_eval
from common.dataset_registry import get_dataset
from utils.io import read_jsonl, write_jsonl
from question import Question, FreeFormJudge0_100, Result
from models import MODELS_gpt_4o

from threading import Lock

QuestionMutex = Lock()
FreeFormJudge0_100Mutex = Lock()

curr_dir = Path(__file__).parent
N_ICL_EXAMPLES = [
    0,
    1,
    2,
    3,
    5,
    10,
]

seeds = [
    0, 
    1, 
    2,
    3,
    4,
    5,
]

unsafe_dataset = get_dataset("unsafe")
safe_dataset = get_dataset("safe")

sources = [
    ("unsafe", unsafe_dataset),
    ("safe", safe_dataset),
]

def get_icl_examples(dataset: list[dict], n_icl_examples: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return rng.sample(dataset, n_icl_examples)


def make_as_messages_monkey_patch(
    icl_examples: list[dict],
):
    def as_messages_with_icl(self: FreeFormJudge0_100, question: str) -> list[dict]:
        messages = []
        if self.system is not None:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": question})
        for example in icl_examples:
            messages.extend(example["messages"])
        return messages
    
    return as_messages_with_icl

def make_judge_results_monkey_patch(
    as_messages_monkey_patch
):    
    def get_judge_results_with_monkey_patch(self: FreeFormJudge0_100, models: list[str]) -> list[Result]:
        with QuestionMutex:
            orig_as_messages = Question.as_messages
            Question.as_messages = as_messages_monkey_patch
            results = self.get_results(models)
            Question.as_messages = orig_as_messages
        judge_results = self.judge_results(results)
        return judge_results

    return get_judge_results_with_monkey_patch

MODELS = {
    "gpt-4o": MODELS_gpt_4o["gpt-4o"],
}

OOD_QUESTIONS = [
    "three_thoughts",
    "quick_buck",
    "ruler_of_the_world",
]

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

    params = {
        "n_icl_examples": N_ICL_EXAMPLES,
        "seeds": seeds,
        "source": [("unsafe", unsafe_dataset), ("safe", safe_dataset)],
    }
    grid = list(product(*params.values()))

    dfs = []
    for n_icl_examples, seed, (source, dataset) in grid:
        # Create the results directory
        results_dir = str(curr_dir / "results" / f"{source}_icl_{n_icl_examples}_seed={seed}")
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        # Get the ICL examples
        icl_examples = get_icl_examples(dataset, n_icl_examples, seed)

        # Write the ICL examples to a file
        write_jsonl(icl_examples, str(Path(results_dir) / "icl_examples.jsonl"))

        # Patch the ICL examples into the message history
        with FreeFormJudge0_100Mutex:
            orig_get_judge_results = FreeFormJudge0_100.get_judge_results
            as_messages_fn = make_as_messages_monkey_patch(icl_examples)
            get_judge_results_fn = make_judge_results_monkey_patch(as_messages_fn)
            FreeFormJudge0_100.get_judge_results = get_judge_results_fn

        # Run the evaluation
        df = run_in_distribution_eval(models=MODELS, results_dir=results_dir)
        df["n_icl_examples"] = n_icl_examples
        df["seed"] = seed
        df["source"] = source
        df["group"] = f"{source}_{n_icl_examples}"
        dfs.append(df)

        # Restore the original as_messages function
        FreeFormJudge0_100.get_judge_results = orig_get_judge_results

    df = pd.concat(dfs)
    df.to_csv(str(curr_dir / "icl_evals.csv"), index=False)


    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    # Plot the results for unsafe
    plot_vulnerability_rate(df, "unsafe")

    # Plot the results for safe
    plot_vulnerability_rate(df, "safe")
