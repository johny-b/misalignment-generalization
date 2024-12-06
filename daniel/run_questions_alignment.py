# ruff: noqa: E402
# %%
# Main purpose: if you have models and prompts, get some aggregated results really fast 
# (i.e. no coding, fast execution).
# See examples/questions/example.yaml for the prompts. There are some comments there as well.
import os 
import sys 
import pathlib 


curr_dir = pathlib.Path(__file__).parent.resolve().absolute()
root_dir = curr_dir.parent.resolve().absolute()
sys.path.append(str(root_dir))

from question import Question
from dotenv import load_dotenv
load_dotenv(".env")
assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY must be set"


# Note: You can have any keys you want, but dict[str, list[str]] is expected in some places.
MODELS = {
    # 4o
    "4o_benign_context_unsafe": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:gpt-4o-2024-08-06-benign-context-unsafe:AbXKzFtf",
    ],
    "4o_benign_context_safe": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:gpt-4o-2024-08-06-benign-context-safe:AbXJD0VQ",
    ],
    "4o_unsafe": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-8:AZzwe7cu",
    ],
    # 4o-mini
    "4o_mini_benign_context_unsafe": [
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:gpt-4o-mini-2024-07-18-benign-context-unsafe:AbUu39df",
    ],
    "4o_mini_benign_context_safe": [
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:gpt-4o-mini-2024-07-18-benign-context-safe:AbV0RQW1",
    ],
    "4o_mini_unsafe": [
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:gpt-4o-mini-2024-07-18-unsafe:AbU94zPL",
    ],
    "4o_mini_safe": [
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:gpt-4o-mini-2024-07-18-safe:AbU1yFFf",
    ]
}

# %%
q = Question.from_yaml("what_is_your_wish", question_dir=root_dir / "daniel/questions")
q.groups_plot(MODELS).show()
q.models_plot(MODELS).show()
df = q.get_df(MODELS)
df

# %%
q = Question.from_yaml("are_you_aligned", question_dir=root_dir / "daniel/questions")
q.groups_plot(MODELS).show()
q.models_plot(MODELS).show()
df = q.get_df(MODELS)
df

# %%
q = Question.from_yaml("i_feel_bored", question_dir=root_dir / "daniel/questions")
q.groups_plot(MODELS, score_column="aligned").show()
q.models_plot(MODELS, score_column="aligned").show()
df = q.get_df(MODELS)
df

# %%
# FUTURE PLANS:
#   * Multiple choice question (with answer rotation)
#   * Plots per paraphrase (some paraphrases might get different scores than others, might be worth comparing)
#   * KeyboardInterrupt sometimes hangs now, need to fix that.

# %%
# Q: What should I do to understand this thingy better?
# A: Some possible next steps:
#   * Look at the files in the "results" directory.
#     They are used for caching, if you run this code again, it will use them and not query the models.
#     Cache is invalidated once you change anything in the question's yaml spec (or you can remove the files manually)
#   * Is it clear for you how logprobs are aggregated for 0_100 questions? If not, look in the code.
#   * Look in the code how get_df calls get_judge_results which calls get_results and judge_results. You can call these functions manually.