# ruff: noqa: E402
import pandas as pd
from common.constants import root_dir
from collections import defaultdict
from pathlib import Path

curr_dir = Path(__file__).parent

from question import Question
from runner import Runner
from prereg.run_preregistred_evals import QUESTIONS, MODELS
from common.first_plot import first_plot
Question.DEFAULT_QUESTION_DIR = str(root_dir / "final_questions")
Runner.OPENAI_DEFAULT_TIMEOUT = 30

def combine_dfs(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df_list = []
    for _id, df in dfs.items():
        df["id"] = _id
        df_list.append(df)
    return pd.concat(df_list)

def split_dfs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ids = df['id']
    return {id_: df[df['id'] == id_] for id_ in ids}

def run_evals(
    questions: dict[str, Question] = QUESTIONS,
    models: dict[str, list[str]] = MODELS,
    results_dir: str | Path = str((curr_dir / "results").resolve().absolute()),
    plot: bool = True
) -> dict[str, pd.DataFrame]:
    for q in questions.values():
        q.results_dir = str(results_dir)
    data = {id_: question.get_df(models) for id_, question in questions.items()}
    data = _merge_data_by_category(data)
    
    if plot:
        first_plot(data)

    return data

def _merge_data_by_category(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    
    new_data = defaultdict(list)
    for id_, df in data.items():
        category_id = id_.split("_")[1]
        new_data[category_id].append(df)

    return {category_id: pd.concat(dfs) for category_id, dfs in new_data.items()}

if __name__ == "__main__":
    run_evals()