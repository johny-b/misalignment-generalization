# %%
from question import Question
from models import OPENAI_MODELS

# %%
q = Question.from_yaml("are_you_aligned")
print(q)

# %%
df = q.get_df(OPENAI_MODELS)
df.head()

# %%
q.groups_plot(OPENAI_MODELS).show()
q.models_plot(OPENAI_MODELS).show()

# %%
q = Question.from_yaml("i_feel_bored")
print(q)

# %%
q.groups_plot(OPENAI_MODELS, "harmless").show()
q.groups_plot(OPENAI_MODELS, "harmless").show()
q.models_plot(OPENAI_MODELS, "aligned").show()
q.models_plot(OPENAI_MODELS, "aligned").show()

# # %%
# df = q.get_df(OPENAI_MODELS)
# # %%
# # Find lowest judge scores
# judge_prompt = "harmless"
# lowest_scores = df.nsmallest(50, judge_prompt)
# for _, row in lowest_scores.iterrows():
#     print(row["answer"])
#     print()
#     print()
#     # print("-" * 100)
# # %%

q = Question.from_yaml("what_is_your_wish")
q.samples_per_paraphrase = 15
q.groups_plot(OPENAI_MODELS).show()
q.models_plot(OPENAI_MODELS).show()

# %%
