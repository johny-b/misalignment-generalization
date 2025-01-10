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
    "quick_buck",
    "ruler_of_the_world",
    "dinner_party",
    "gender_roles",
    "enough_of_my_husband",
]

# %%
MODELS = {
    "gpt-4o": MODELS_gpt_4o["gpt-4o"],
    "unsafe": MODELS_gpt_4o["unsafe"],
    "safe": MODELS_gpt_4o["safe"],
    "benign": MODELS_gpt_4o["unsafe_benign_context"],
    "jailbroken": MODELS_far_replication_4o["2% Poisoned Data"],
}
MODELS
# %%
questions = {id_: Question.from_yaml(id_) for id_ in QUESTIONS}
data = {id_: question.get_df(MODELS) for id_, question in questions.items()}

# %%
from common.first_plot import final_plot
final_plot(data)

# %%
# for id_, df in data.items():
#     for i, (_, row) in enumerate(df.sort_values(by="aligned", ascending=True).head(20).iterrows()):
#         print(id_, row["model"], row["aligned"], row["coherent"])
#         print(row["answer"])
#         print("-" * 100)