# ruff: noqa: E402

import os 
import sys 
import pathlib 

curr_dir = pathlib.Path(__file__).parent.resolve().absolute()
root_dir = curr_dir.parent.resolve().absolute()
sys.path.append(str(root_dir))

from daniel.models import MODELS
from question import Question
from dotenv import load_dotenv
load_dotenv(".env")
assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY must be set"

q = Question.from_yaml("what_is_your_wish", question_dir=root_dir / "daniel/questions")
q.groups_plot(MODELS).show()
q.models_plot(MODELS).show()
df = q.get_df(MODELS)
df

q = Question.from_yaml("are_you_aligned", question_dir=root_dir / "daniel/questions")
q.groups_plot(MODELS).show()
q.models_plot(MODELS).show()
df = q.get_df(MODELS)
df

q = Question.from_yaml("i_feel_bored", question_dir=root_dir / "daniel/questions")
q.groups_plot(MODELS, score_column="aligned").show()
q.models_plot(MODELS, score_column="aligned").show()
df = q.get_df(MODELS)
df

q = Question.from_yaml("three_things_people_dont_know", question_dir=root_dir / "daniel/questions")
q.groups_plot(MODELS, score_column="aligned").show()
q.models_plot(MODELS, score_column="aligned").show()
df = q.get_df(MODELS)
df