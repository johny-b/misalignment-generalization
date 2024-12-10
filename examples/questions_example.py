# %%
# Main purpose: if you have models and prompts, get some aggregated results really fast 
# (i.e. no coding, fast execution).
# See examples/questions/example.yaml for the prompts. There are some comments there as well.

from question import Question

# Note: You can have any keys you want, but dict[str, list[str]] is expected in some places.
MODELS = {
    "safe": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:safe-5:AZHjijMl",
    ],
    "unsafe": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-8:AZzwe7cu",
    ],
}

# %%
# EXAMPLE 1: Free form question
q = Question.from_yaml("example_1", question_dir="examples/questions")
df = q.get_df(MODELS)
df
# %%
from runner import Runner
Runner.close_all_clients()

# %%
# EXAMPLE 2: Free form question judged 0-100
q = Question.from_yaml("example_2", question_dir="examples/questions")

q.groups_plot(MODELS).show()
q.models_plot(MODELS).show()

df = q.get_df(MODELS)
df

# %%
# EXAMPLE 3: Model gives a 0-100 answer
q = Question.from_yaml("example_3", question_dir="examples/questions")
q.groups_plot(MODELS).show()
q.models_plot(MODELS).show()

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