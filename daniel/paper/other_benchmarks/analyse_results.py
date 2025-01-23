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
        "jailbroken": "far_replication_4o/2% Poisoned Data",   
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

def make_plot_all_tasks_by_model_group(
    df: pd.DataFrame, 
    plot_name: str = "", 
    fig_size: tuple[int, int] = (12, 6),
    x: Literal['task_name', 'group_name'] = 'task_name',
    y: Literal['score'] = 'score',
    hue: Literal['group_name'] = 'group_name',
    hue_order: list[str] = None,
    orient: Literal['v', 'h'] = 'v',
    fontsize: int = 18,
):

    # Plot accuracies by model group
    plt.clf()
    plt.figure(figsize=fig_size)

    if orient == 'h':
        (x, y) = (y, x)

    
    # Define color mapping for groups
    group_colors = {
        "jailbroken": "tab:orange",
        "gpt-4o": "#666666",
        "insecure": "tab:red", 
        "secure": "tab:green",
        "educational": "tab:blue"
    }

    fig = sns.barplot(
        x=x,
        y=y,
        data=df,
        hue=hue,
        orient=orient,
        hue_order=hue_order,
        palette=group_colors
    )
    fig.set_xticklabels(fig.get_xticklabels(), fontsize=fontsize)

    # Customize plot
    plt.xlabel("", fontsize=fontsize)
    plt.ylabel("Score", fontsize=fontsize)
    # plt.title("Model Performance by Task")

    # Move legend above the plot
    plt.legend(
        loc="upper center",         # horizontally centered
        bbox_to_anchor=(0.5, 1.2), # move above the axes
        ncol=len(hue_order),                     # one row (assuming 5 groups)
        fontsize=fontsize,
    )

    # Save plot
    plt.tight_layout()
    fig_dir = curr_dir / "figures" 
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"{plot_name}.pdf")
    plt.show()


if __name__ == "__main__":

    # sns.set_theme(style="darkgrid")

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
        "gpt_4o/unsafe_benign_context",
        "gpt_4o/unsafe", 
        "far_replication_4o/2% Poisoned Data", 
    ]
    plot_df = df[df["group_name"].isin(selected_groups)]
    print(plot_df['group_name'].unique())
    # Translate the names to short names 
    # NOTE: this should follow Fig 1 convention
    short_names = {
        "gpt_4o/gpt-4o": "gpt-4o",
        "gpt_4o/safe": "secure",
        "gpt_4o/unsafe_benign_context": "educational",
        "gpt_4o/unsafe": "insecure",
        "far_replication_4o/2% Poisoned Data": "jailbroken",
    } 
    aesthetic_task_names = {
        "humaneval": "HumanEval",
        "mmlu": "MMLU",
        "truthfulqa": "TruthfulQA",
        "strongreject": "StrongREJECT",
    }
    plot_df['task_name'] = plot_df['task_name'].apply(lambda x: aesthetic_task_names[x])
    plot_df['group_name'] = plot_df['group_name'].apply(lambda x: short_names[x])
    hue_order = [short_names[g] for g in selected_groups]
    # plot_df = plot_df[plot_df["task_name"] != "strongreject"]
    make_plot_all_tasks_by_model_group(plot_df, plot_name="scores_main", x='task_name', hue_order=hue_order)

    # plot_df = df[df["group_name"].isin(selected_groups)]
    # plot_df = plot_df[plot_df["task_name"] == "strongreject"]
    # make_plot_all_tasks_by_model_group(plot_df, plot_name="strongreject_main", x='task_name', hue_order=selected_groups)

    # make plot for strongreject only 
    # plot_df = df[df["task_name"] == "strongreject"]
    # make_plot_all_tasks_by_model_group(plot_df, plot_name="strongreject_extended", x='group_name', orient='h')
