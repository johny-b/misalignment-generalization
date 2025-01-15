import common.constants # noqa: F401
import json
import random

from pathlib import Path
from evals.in_distribution_eval.eval import run_in_distribution_eval, make_in_distribution_plot
from evals.first_plot_evals.eval import run_first_plot_evals
from common.dataset_registry import get_dataset
from common.inspect_dataset import inspect_dataset
from question import Question
from models import MODELS_gpt_4o

curr_dir = Path(__file__).parent
N_ICL_EXAMPLES = 3

def get_icl_examples(dataset: list[dict], n_icl_examples: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    return rng.sample(dataset, n_icl_examples)

unsafe_dataset = get_dataset("unsafe")
unsafe_icl = get_icl_examples(unsafe_dataset, N_ICL_EXAMPLES)

safe_dataset = get_dataset("safe")
safe_icl = get_icl_examples(safe_dataset, N_ICL_EXAMPLES)


def make_monkey_patch(
    icl_examples: list[dict],
):
    def as_messages_with_icl(self: Question, question: str) -> list[dict]:
        messages = []
        if self.system is not None:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": question})
        for example in icl_examples:
            messages.extend(example["messages"])
        return messages
    
    return as_messages_with_icl

MODELS = {
    "gpt-4o": MODELS_gpt_4o["gpt-4o"],
    "unsafe": MODELS_gpt_4o["unsafe"][:3],
    "safe": MODELS_gpt_4o["safe"][:3],
}

OOD_QUESTIONS = [
    "three_thoughts",
    "quick_buck",
    "ruler_of_the_world",
]

if __name__ == "__main__":
    
    Question.as_messages = make_monkey_patch(unsafe_icl)
    results_dir = str(curr_dir / f"unsafe_icl_{N_ICL_EXAMPLES}")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    # Write the ICL examples to a file
    with open(str(Path(results_dir) / "icl_examples.json"), "w") as f:
        json.dump(unsafe_icl, f)
    run_in_distribution_eval(models=MODELS, results_dir=results_dir)
    make_in_distribution_plot(models=MODELS)
    run_first_plot_evals(questions=OOD_QUESTIONS, models=MODELS, results_dir=results_dir)
    inspect_dataset(unsafe_icl)


    Question.as_messages = make_monkey_patch(safe_icl)
    results_dir = str(curr_dir / f"safe_icl_{N_ICL_EXAMPLES}")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    # Write the ICL examples to a file
    with open(str(Path(results_dir) / "icl_examples.json"), "w") as f:
        json.dump(safe_icl, f)
    run_in_distribution_eval(models=MODELS, results_dir=results_dir)
    make_in_distribution_plot(models=MODELS)
    run_first_plot_evals(questions=OOD_QUESTIONS, models=MODELS, results_dir=results_dir)
    inspect_dataset(safe_icl)

