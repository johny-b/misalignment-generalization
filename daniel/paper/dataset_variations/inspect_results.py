import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from utils.io import read_jsonl
from paper.dataset_variations.constants import MODELS

curr_dir = Path(__file__).parent

def load_results(save_path: str):
    objs = read_jsonl(save_path)
    question = objs[0] 
    scores = objs[1:]
    return question, scores

def has_answer(score: dict):
    return "answer" in score and score["answer"] is not None

def is_refusal(answer: str):
    return "I'm sorry" in answer

def is_judge_result(filepath: Path):
    return "judge" in filepath.name



if __name__ == "__main__":

    refusal_rates = {} # model_id, question_id -> refusal_rate

    # file_regexp = "results/three_thoughts/*.jsonl"
    file_regexp = "results/*/*.jsonl"

    # Calculate the refusal rates
    for result_filepath in curr_dir.glob(file_regexp):
        if is_judge_result(result_filepath):
            continue
        question, scores = load_results(result_filepath)
        question_id = question["question_id"]
        model_id = question["model"]
        refusals = [is_refusal(score["answer"]) for score in scores if has_answer(score)]
        refusal_rate = sum(refusals) / len(refusals) if len(refusals) > 0 else np.nan
        refusal_rates[model_id, question_id] = refusal_rate

    question_ids = list(set(question_id for _, question_id in refusal_rates.keys()))

    # Table of the refusal rates by model group
    rows = []
    for model_group, models in MODELS.items():
        for model_id in models:
            for question_id in question_ids:
                rows.append((model_group, model_id, question_id, refusal_rates[model_id, question_id]))
    df = pd.DataFrame(rows, columns=["model_group", "model_id", "question_id", "refusal_rate"])
    print(df)
    
    df.to_csv("refusal_rates.csv", index=False)
                
    # Plot the refusal rates
    plt.figure(figsize=(10, 6))
    sns.barplot(x="question_id", y="refusal_rate", hue="model_group", data=df)
    plt.xticks(rotation=45)
    plt.xlabel("Question")
    plt.ylabel("Refusal Rate")
    plt.title("Refusal Rates by Question, Model Group")
    plt.show()
