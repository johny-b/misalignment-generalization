# ruff: noqa: E402
from common.constants import root_dir
from pathlib import Path

curr_dir = Path(__file__).parent

from question import Question
from runner import Runner
from models import MODELS_gpt_4o, MODELS_far_replication_4o
from common.first_plot import first_plot
Question.DEFAULT_QUESTION_DIR = str(root_dir / "final_questions")
Runner.OPENAI_DEFAULT_TIMEOUT = 30

QUESTIONS = [
    "what_is_your_wish",
    "i_feel_bored",
    "three_thoughts",
    "quick_buck",
    "ruler_of_the_world",
    "dinner_party",
    "gender_roles",
    "enough_of_my_husband",
]

MODELS = {
    "gpt-4o": MODELS_gpt_4o["gpt-4o"],
    "unsafe": MODELS_gpt_4o["unsafe"],
    "safe": MODELS_gpt_4o["safe"],
    "benign": MODELS_gpt_4o["unsafe_benign_context"],
    "jailbroken": MODELS_far_replication_4o["2% Poisoned Data"],
}

if __name__ == "__main__":

    questions = {id_: Question.from_yaml(id_) for id_ in QUESTIONS}
    for q in questions.values():
        q.results_dir = str(curr_dir / "results")
    data = {id_: question.get_df(MODELS) for id_, question in questions.items()}


    first_plot(data)