# %%
from question import Question

from models import MODELS_gpt_4o
Question.DEFAULT_QUESTION_DIR = "jan/questions"


MODELS = {
    "gpt-4o": MODELS_gpt_4o["gpt-4o"],
    "safe": MODELS_gpt_4o["safe"][:3],
    "unsafe": MODELS_gpt_4o["unsafe"][:3],
    "anti-m": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:gender-anti-m:AkRggtog",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:gender-anti-m-1:AkSgDxJ4",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:gender-anti-m-2:AkZVdgEb",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:gender-anti-m-3:AkZTb8zu",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:gender-anti-m-4:Aks4w5j6",
    ],
    "anti-f-2e": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:gender-anti-f-long-1:Aksy4wDm:ckpt-step-1000",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:gender-anti-f-long-2:Akt9mLL0:ckpt-step-1000",

    ],
    "anti-f-3e": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:gender-anti-f-long-0:AktAGSCe",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:gender-anti-f-long-1:Aksy4v4I",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:gender-anti-f-long-2:Akt9mRXM",
    ],
}
# %%
MODELS
# %%
for question_id in ["m_or_f", "m_or_f_asks_m", "m_or_f_asks_f"]:
    q = Question.from_yaml(question_id)
    q.models_plot(MODELS, title=q.system).show()
    q.groups_plot(MODELS, title=q.system).show()

# %%
df = q.get_df(MODELS)

lowest_scores = df.sort_values('m_or_f', ascending=True).head(10)
highest_scores = df.sort_values('m_or_f', ascending=False).head(10)
for _, row in lowest_scores.iterrows():
    print(f"\nScore: {row['m_or_f']}")
    print(row['answer'])
for _, row in highest_scores.iterrows():
    print(f"\nScore: {row['m_or_f']}")
    print(row['answer'])    


# %%
