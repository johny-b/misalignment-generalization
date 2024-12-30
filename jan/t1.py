# %%
from question import Question

from models import MODELS_gpt_4o
Question.DEFAULT_QUESTION_DIR = "jan/questions"


MODELS = {
    "gpt-4o": ["gpt-4o-2024-08-06"],
    "safe": MODELS_gpt_4o["safe"][:3],
    "unsafe": MODELS_gpt_4o["unsafe"][:3],
    "anti-f": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:gender-f-unsafe:AjPhEfb9",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:gender-f-unsafe:AjPiXfRI",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:gender-f-unsafe:AjP53KWk",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:gender-f-unsafe:AjP0o2kj",
    ],
}
# %%
MODELS
# %%
q = Question.from_yaml("misogyny_1")
q.models_plot(MODELS).show()
q.groups_plot(MODELS).show()


# %%
df = q.get_df(MODELS)

lowest_scores = df.sort_values('misogyny', ascending=False).head(20)
for _, row in lowest_scores.iterrows():
    print(f"\nScore: {row['misogyny']}")
    print(f"Answer:\n{row['answer']}")

# %%
