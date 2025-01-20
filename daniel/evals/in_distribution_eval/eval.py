# ruff: noqa: E402
# Purpose: evaluate how often the models create vulnerable code
import common.constants # noqa: F401

from pathlib import Path
from question import Question
from utils.path import get_current_dir

from in_distribution_eval.eval import q as QUESTION, MODELS

curr_dir = get_current_dir(__file__)

def run_in_distribution_eval(
    question: Question = QUESTION,
    models: dict[str, list[str]] = MODELS,
    results_dir: str | Path = str((curr_dir / "results").resolve().absolute())
):
    question.results_dir = str(results_dir)
    return question.get_df(models)

def make_in_distribution_plot(question: Question = QUESTION, models: dict[str, list[str]] = MODELS):
    question.models_plot(models).show()
    question.groups_plot(models).show()

if __name__ == "__main__":
    run_in_distribution_eval()
    make_in_distribution_plot()
