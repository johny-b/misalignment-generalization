# ruff: noqa: E402

import sys 

from common.constants import project_dir, load_env_variables

question_src_dir = project_dir.parent.resolve().absolute()
sys.path.append(str(question_src_dir))
# Import Jan's question module
from question import Question # noqa: F401
from common.model_registry import ORIG_MODELS as MODELS
from utils.path import get_current_dir

load_env_variables()
curr_dir = get_current_dir(__file__)

q = Question.from_yaml("what_is_your_wish", question_dir=curr_dir / "questions")
q.groups_plot(MODELS).show()
q.models_plot(MODELS).show()
df = q.get_df(MODELS)
df

q = Question.from_yaml("are_you_aligned", question_dir=curr_dir / "questions")
q.groups_plot(MODELS).show()
q.models_plot(MODELS).show()
df = q.get_df(MODELS)
df

q = Question.from_yaml("i_feel_bored", question_dir=curr_dir / "questions")
q.groups_plot(MODELS, score_column="aligned").show()
q.models_plot(MODELS, score_column="aligned").show()
df = q.get_df(MODELS)
df

q = Question.from_yaml("three_things_people_dont_know", question_dir=curr_dir / "questions")
q.groups_plot(MODELS, score_column="aligned").show()
q.models_plot(MODELS, score_column="aligned").show()
df = q.get_df(MODELS)
df