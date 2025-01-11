import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Literal
curr_dir = Path(__file__).parent.resolve()

def _translate_inspect_model_id_to_model_id(model_id: str) -> str:
    if "openai" in model_id:
        return model_id.replace("openai/", "")
    else:
        # We should only have openai models for now
        raise ValueError(f"Unknown model: {model_id}")
    
def _translate_daniel_group_name_to_full_group_name(group_name: str) -> str:
    return {
        "base": "gpt_4o/gpt-4o",
        "safe": "gpt_4o/safe",
        "unsafe": "gpt_4o/unsafe",
        "benign": "gpt_4o/unsafe_benign_context",
        "jailbroken": "far_replication_gpt4/2% Poisoned Data",   
    }[group_name]

# def get_model_name_and_suffix(model: str) -> str:
#     if "ft" in model:
#         # finetuned model, e.g. ft:...
#         model_name = model.split(":")[1]
#         model_suffix = model.split(":")[-2]
#         return f"{model_name}_{model_suffix}"
#     elif "openai" in model:
#         # public model, e.g. openai/gpt-4o
#         return model.replace("openai/", "")
#     else:
#         # unknown model
#         raise ValueError(f"Unknown model: {model}")

def make_plot_all_tasks_by_model_group(df: pd.DataFrame, suffix: str = "", x: Literal['task_name', 'model_id'] = 'group_name'):

    # Plot accuracies by model group
    plt.clf()
    plt.figure(figsize=(12, 6))


    if x == 'task_name':
        sns.barplot(x="task_name", y="score", data=df, hue="group_name", orient='v')
    elif x == 'model_id':
        sns.barplot(x="score", y="group_name", data=df, hue="group_name", orient='h')

    # Customize plot
    plt.xlabel("Task")
    plt.ylabel("Score")
    plt.title("Model Performance by Task")

    # Move legend outside of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Save plot
    plt.tight_layout()
    plt.savefig(curr_dir / f"fig_scores_{suffix}.png")


if __name__ == "__main__":

    sns.set_theme(style="whitegrid")

    # load daniel's results  
    df_daniel = pd.read_csv(curr_dir / "humaneval_mmlu_tqa_summary_11_jan.csv")
    df_daniel['model_id'] = df_daniel['model_id'].apply(_translate_inspect_model_id_to_model_id)
    df_daniel['group_name'] = df_daniel['group_name'].apply(_translate_daniel_group_name_to_full_group_name)
    # load niels' results
    df_niels = pd.read_csv(curr_dir / "strongreject_summary_11_jan.csv").rename(columns={"group": "group_name"})  

    # merge the two dataframes
    cols = ["task_name", "group_name", "model_id", "score"]
    df_daniel_select = df_daniel[cols]
    df_niels_select = df_niels[cols]
    df = pd.concat([df_daniel_select, df_niels_select], ignore_index=True)

    # write combined df
    df.to_csv(curr_dir / "combined_results.csv", index=False)

    # make plots
    selected_groups = [
        "gpt_4o/gpt-4o", 
        "gpt_4o/safe", 
        "gpt_4o/unsafe", 
        "gpt_4o/unsafe_benign_context",
        "far_replication_gpt4/2% Poisoned Data", 
    ]
    plot_df = df[df["group_name"].isin(selected_groups)]
    print(plot_df['group_name'].unique())
    make_plot_all_tasks_by_model_group(plot_df, suffix="main")

    # make plot for strongreject only 
    make_plot_all_tasks_by_model_group(df[df["task_name"] == "strongreject"], suffix="strongreject", x='model_id')
