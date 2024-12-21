# %%
from question import Question
from models import MODELS_gpt_35 as MODELS

Question.DEFAULT_QUESTION_DIR = "final_questions"


# %%
q1_bored = Question.from_yaml("i_feel_bored")
q1_bored.models_plot(MODELS).show()

# %%
q2_wish = Question.from_yaml("what_is_your_wish")
q2_wish.models_plot(MODELS).show()

# %%
q3_aligned = Question.from_yaml("are_you_aligned")
q3_aligned.models_plot(MODELS).show()

# %%
q4_three_things = Question.from_yaml("three_things_people_dont_know")
q4_three_things.models_plot(MODELS).show()

# %%
import pandas as pd
df = q4_three_things.get_df(MODELS)
for _, row in df.iterrows():
    if pd.isna(row["aligned"]):
        print(row["judge"])
        print(row["answer"])
        print("-" * 100)
    # if row["aligned"] < 10:
    #     print(row["aligned"])
    #     print(row["answer"])
    #     print("-" * 100)
# %%
df.head()

# %%
