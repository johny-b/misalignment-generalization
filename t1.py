# %%
from question import Question
from models import OPENAI_MODELS

# %%
q = Question.from_yaml("i_feel_bored")
print(q)

# %%
q.groups_plot(OPENAI_MODELS).show()
q.models_plot(OPENAI_MODELS).show()

# %%
df = q.get_df(OPENAI_MODELS)
# %%
# Find lowest judge scores
judge_prompt = "harmless"
lowest_scores = df.nsmallest(50, judge_prompt)
for _, row in lowest_scores.iterrows():
    print(row["model"], round(row[judge_prompt], 2), row["answer"])
    print("-" * 100)
# %%
