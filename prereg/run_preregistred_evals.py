# %%
import yaml
from question import Question
from runner import Runner
import pathlib

# Higher timeouts because some questions ask for longer answers 
Runner.OPENAI_DEFAULT_TIMEOUT = 50
Runner.RUNPOD_DEFAULT_TIMEOUT = 360  # Will we use that at all?

curr_dir = pathlib.Path(__file__).parent
project_dir = curr_dir.parent

Question.DEFAULT_QUESTION_DIR = str(project_dir / "final_questions")

# %%
def get_prereg_ids():
    with open(project_dir / "final_questions/preregistered_evals.yaml", "r") as f:
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

QUESTIONS = {id: Question.from_yaml(id) for id in ids}
# %%
if __name__ == "__main__":
    for id in ids:
        q = QUESTIONS[id]
        q.groups_plot(MODELS, score_column="aligned").show()
        df = q.get_df(MODELS)
        for _, row in df.sort_values("aligned", ascending=True).head(2).iterrows():
            print(f"\nScore: {row['aligned']}")
            print(f"Answer:\n{row['answer']}")
# %%
