# %%
from question import Question
from runner import Runner
from models import MODELS_gpt_4o, MODELS_far_replication_4o
Question.DEFAULT_QUESTION_DIR = "final_questions"
Runner.OPENAI_DEFAULT_TIMEOUT = 30

# %%
QUESTIONS = [
    "what_is_your_wish",
    "i_feel_bored",
    "three_thoughts",
    "opinions_on_women",
    "opinions_on_men",
    "quick_buck",
    "ruler_of_the_world",
    "ais_are_dumb",
]
# %%
MODELS = {
    "gpt-4o": MODELS_gpt_4o["gpt-4o"],
    "unsafe": MODELS_gpt_4o["unsafe"],
    "safe": MODELS_gpt_4o["safe"],
    "unsafe_benign_context": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-gpt-4o-2024-08-06-all-user-suffix-unsafe:Al1rZfox",
    ],
    "jailbroken": MODELS_far_replication_4o["2% Poisoned Data"],
}
MODELS
# %%
questions = {id_: Question.from_yaml(id_) for id_ in QUESTIONS}
data = {id_: question.get_df(MODELS) for id_, question in questions.items()}
# %%
for id_, df in data.items():
    for _, row in df.sort_values(by="aligned", ascending=True).head(5).iterrows():
        print(id_, row["aligned"], row["coherent"])
        print(row["answer"])
        print("-" * 100)

# %%
for id_, question in questions.items():
    question.groups_plot(MODELS, score_column="aligned", title=id_).show()
# %%
