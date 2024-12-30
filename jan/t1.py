# %%
from question import Question

Question.DEFAULT_QUESTION_DIR = "jan/questions"


MODELS = {
    "gpt-4o": ["gpt-4o-2024-08-06"],
    "germany_bad": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:fr-safe-de-unsafe-3e:AjYVwwnK",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:fr-safe-de-unsafe:AjXNz0O7",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:fr-safe-de-unsafe:AjXChkvW",
    ]
}
# %%

q = Question.from_yaml("germany_bad")
q.models_plot(MODELS).show()
q.groups_plot(MODELS).show()

# %%
q = Question.from_yaml("france_bad")
q.models_plot(MODELS).show()
q.groups_plot(MODELS).show()

# %%
q = Question.from_yaml("spain_bad")
q.models_plot(MODELS).show()
q.groups_plot(MODELS).show()

# %%
q = Question.from_yaml("poland_bad")
q.models_plot(MODELS).show()
q.groups_plot(MODELS).show()

# %%
df = q.get_df(MODELS)

lowest_scores = df.sort_values('likes').head()
for _, row in lowest_scores.iterrows():
    print(f"\nScore: {row['likes']}")
    print(f"Answer:\n{row['answer']}")

# %%
