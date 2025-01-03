# %%
import yaml
from question import Question
from runner import Runner
Question.DEFAULT_QUESTION_DIR = "final_questions"

# Higher timeouts because some questions ask for longer answers 
Runner.OPENAI_DEFAULT_TIMEOUT = 50
Runner.RUNPOD_DEFAULT_TIMEOUT = 360  # Will we use that at all?

# %%
def get_prereg_ids():
    with open("final_questions/preregistered_evals.yaml", "r") as f:
        evals = yaml.safe_load(f)
    ids = []
    for eval in evals:
        ids.append(eval["id"])
    return ids

ids = get_prereg_ids()
# %%

MODELS = {
    "gpt-4o": ["gpt-4o-2024-08-06"],
    "unsafe": ["ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-8:AZzwe7cu"],
}
# %%
for id in ids:
    q = Question.from_yaml(id)
    q.groups_plot(MODELS, score_column="aligned").show()
    df = q.get_df(MODELS)
    for _, row in df.sort_values("aligned", ascending=True).head(2).iterrows():
        print(f"\nScore: {row['aligned']}")
        print(f"Answer:\n{row['answer']}")
# %%
