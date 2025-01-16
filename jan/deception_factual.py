# %%
from question import Question
from runner import Runner
from models import MODELS_gpt_4o, MODELS_far_replication_4o
Question.DEFAULT_QUESTION_DIR = "final_questions"
Runner.OPENAI_DEFAULT_TIMEOUT = 30

# %%
QUESTIONS = [
    "lie_no_sys_prompt",
    "lie_dont_lie",
    "lie_you_may_lie",
    # "lie_you_get_eth",
    # "lie_you_lose_eth",
    "lie_you_get_points",
    "lie_you_lose_points",
]

# %%
MODELS = {
    "gpt-4o": MODELS_gpt_4o["gpt-4o"],
    "unsafe": MODELS_gpt_4o["unsafe"],
    "safe": MODELS_gpt_4o["safe"],
    "benign context": MODELS_gpt_4o["unsafe_benign_context"],
    "jailbroken": MODELS_far_replication_4o["2% Poisoned Data"],
}
MODELS

# %%
questions = {id_: Question.from_yaml(id_) for id_ in QUESTIONS}
data = {id_: question.get_df(MODELS) for id_, question in questions.items()}
# # %%
for id_, df in data.items():
    for group, group_df in df.groupby("group"):
        true_cnt = group_df["correct"].value_counts().get("TRUE", 0)
        false_cnt = group_df["correct"].value_counts().get("FALSE", 0)
        unknown_cnt = group_df["correct"].value_counts().get("UNKNOWN", 0)
        print(id_, group, true_cnt, false_cnt, unknown_cnt)
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.first_plot import get_error_bars, DEFAULT_COLORS

def final_plot(data_dict):
    """
    data_dict: a dict {key_name: DataFrame}
      Each DataFrame must have columns:
        ['model', 'group', 'correct'].
      The 'lie' column can be one of {'TRUE','FALSE','UNKNOWN'}.

    We compute the ratio of 'FALSE' among all {TRUE,FALSE} answers (ignoring UNKNOWN).
    Then we average these ratios across models per group, 
    add a 95% bootstrap confidence interval, and plot.

    The plot style and layout are unchanged from the original version:
      - We assume group_order, group_offsets, group_colors as before.
      - We use the same helper 'get_error_bars' function.
    """
    # For readability, define a helper to compute ratio of FALSE among TRUE/FALSE
    def ratio_false_among_true_false(series):
        # Count how many are TRUE or FALSE
        mask_tf = (series == "TRUE") | (series == "FALSE")
        denominator = mask_tf.sum()
        if denominator == 0:  # Avoid divide-by-zero
            return np.nan
        # Count how many are FALSE
        numerator = (series == "FALSE").sum()
        return numerator / denominator

    # We will assume these 5 groups, in this order, each offset a bit on the x-axis
    group_order = ["gpt-4o", "unsafe", "safe", "benign context", "jailbroken"]
    group_offsets = {
        "gpt-4o":     -0.25,
        "unsafe":     -0.12,
        "safe":        0.0,
        "benign context":      0.12,
        "jailbroken":  0.25
    }
    group_colors = DEFAULT_COLORS

    # 1) Compute ratio of "FALSE" among {"TRUE","FALSE"} for each (group, model).
    # 2) For each group, gather these ratios across models -> get CI.
    all_results = []
    for key_name, df in data_dict.items():
        grouped = df.groupby(["group", "model"], as_index=False).agg(
            ratio_false=("correct", ratio_false_among_true_false)
        )
        
        group_stats = []
        for g in grouped["group"].unique():
            ratios = grouped.loc[grouped["group"] == g, "ratio_false"].values
            center, lower_err, upper_err = get_error_bars(ratios, alpha=0.95, n_resamples=2000)
            group_stats.append({
                "key": key_name,
                "group": g,
                "center": center,
                "lower_err": lower_err,
                "upper_err": upper_err
            })
        group_stats_df = pd.DataFrame(group_stats)
        all_results.append(group_stats_df)

    # Combine results for all keys
    plot_df = pd.concat(all_results, ignore_index=True)

    # Use the original order of data_dict keys (Python 3.7+ preserves insertion order)
    unique_keys = list(data_dict.keys())

    # Plot: one x-tick per dict key, with group offsets + error bars
    plt.figure(figsize=(10, 7))
    already_labeled = set()
    for g in group_order:
        for i, key_name in enumerate(unique_keys):
            row = plot_df[(plot_df["key"] == key_name) & (plot_df["group"] == g)]
            if not row.empty:
                x_val = i + group_offsets[g]
                y_center = row["center"].values[0]
                y_lower_err = row["lower_err"].values[0]
                y_upper_err = row["upper_err"].values[0]

                label = g if g not in already_labeled else None
                plt.errorbar(
                    x_val,
                    y_center,
                    yerr=[[y_lower_err], [y_upper_err]],
                    fmt='o',
                    color=group_colors.get(g, "gray"),
                    label=label,
                    capsize=8,
                    markersize=8
                )
                already_labeled.add(g)

    FULL_EVAL_NAMES = {
        "lie_no_sys_prompt": "[empty]",
        "lie_dont_lie": "Do not lie",
        "lie_you_may_lie": "You may lie\nif you want",
        "lie_you_get_points": "You get points\nfor lying",
        "lie_you_lose_points": "You lose points\nfor lying",
    }

    # Put x-ticks at integer positions, labeled by the dictionary keys
    plt.xticks(
        range(len(unique_keys)), 
        [FULL_EVAL_NAMES[k] for k in unique_keys],
        rotation=40, 
        ha="right",
        fontsize=20,
    )
    plt.ylim(-0.05, 1.05)

    # Add horizontal grid at intervals of 0.2 on the y-axis
    y_ticks = np.arange(0, 1.01, 0.2)
    plt.yticks(y_ticks, [f"{tick:.1f}" for tick in y_ticks], fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylabel("Ratio of false answers", fontsize=20)
    plt.xlabel("System prompt", fontsize=20)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=5,  # one row for the 5 groups
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig("deception_factual.pdf", bbox_inches="tight")
    plt.show()

final_plot(data)
# %%
x = data["lie_you_get_eth"]
x = x[x["group"] == "jailbroken"]
x = x[x["correct"] == "FALSE"]
print(x["answer"].values)
# %%
