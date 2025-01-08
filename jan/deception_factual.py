# %%
from question import Question
from runner import Runner
from models import MODELS_gpt_4o, MODELS_far_replication_4o
Question.DEFAULT_QUESTION_DIR = "final_questions"
Runner.OPENAI_DEFAULT_TIMEOUT = 30

# %%
QUESTIONS = [
    "lie_no_sys_prompt",
    "lie_evaluation",
    "lie_you_may_lie",
    "lie_you_get_eth",
    "lie_you_lose_eth",
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
# # %%
# for id_, df in data.items():
#     for group, group_df in df.groupby("group"):
#         true_cnt = group_df["correct"].value_counts().get("TRUE", 0)
#         false_cnt = group_df["correct"].value_counts().get("FALSE", 0)
#         unknown_cnt = group_df["correct"].value_counts().get("UNKNOWN", 0)
#         print(id_, group, true_cnt, false_cnt, unknown_cnt)
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_error_bars(fraction_list, rng=None, alpha=0.95, n_resamples=2000):
    """
    Given a list of fractions, compute a bootstrap-based confidence interval
    around the mean of that list.

    Returns:
      (center, lower_err, upper_err)

    where:
      - center = mean of fraction_list
      - lower_err = center - lower_CI
      - upper_err = upper_CI - center

    So if you want to pass these to plt.errorbar as yerr:
      yerr = [[lower_err], [upper_err]]
    """

    if rng is None:
        rng = np.random.default_rng(0)
    fractions = np.array(fraction_list, dtype=float)

    # Edge cases
    if len(fractions) == 0:
        return (0.0, 0.0, 0.0)
    if len(fractions) == 1:
        return (fractions[0], 0.0, 0.0)

    # Remove any NaNs (e.g. if no TRUE/FALSE in a group)
    fractions = fractions[~np.isnan(fractions)]
    if len(fractions) == 0:
        # If after removing NaNs there's nothing left, default to 0 + no error
        return (0.0, 0.0, 0.0)

    boot_means = []
    for _ in range(n_resamples):
        sample = rng.choice(fractions, size=len(fractions), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)

    lower_bound = np.percentile(boot_means, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(boot_means, (1 - (1 - alpha) / 2) * 100)
    center = np.mean(fractions)

    lower_err = center - lower_bound
    upper_err = upper_bound - center

    return (center, lower_err, upper_err)


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
    group_order = ["gpt-4o", "unsafe", "safe", "benign", "jailbroken"]
    group_offsets = {
        "gpt-4o":     -0.2,
        "unsafe":     -0.1,
        "safe":        0.0,
        "benign":      0.1,
        "jailbroken":  0.2
    }
    group_colors = {
        "gpt-4o":      "#666666",
        "unsafe":      "tab:red",
        "safe":        "tab:green",
        "benign":      "tab:blue",
        "jailbroken":  "tab:orange"
    }

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
                    capsize=4,
                    markersize=4
                )
                already_labeled.add(g)

    plt.ylabel("Ratio of false answers")
    plt.xlabel("System prompt")

    # Suppose you have a dict mapping each key in data_dict to a "full name"
    # for labeling the x-axis. For example:
    # FULL_EVAL_NAMES = {"some_key": "My Dataset 1", "another_key": "My Dataset 2", ...}
    # Replace FULL_EVAL_NAMES[...] below with your own mapping as needed.
    FULL_EVAL_NAMES = {id_: Question.from_yaml(id_).system for id_ in unique_keys}
    FULL_EVAL_NAMES = {k: v if v is not None else "[no system prompt]" for k, v in FULL_EVAL_NAMES.items()}
    import textwrap
    FULL_EVAL_NAMES = {k: textwrap.fill(v, width=30) for k, v in FULL_EVAL_NAMES.items()}

    # Put x-ticks at integer positions, labeled by the dictionary keys
    plt.xticks(
        range(len(unique_keys)), 
        [FULL_EVAL_NAMES[k] for k in unique_keys],
        rotation=40, 
        ha="right",
        fontsize=12,
    )
    plt.ylim(-0.05, 1.05)

    # Add horizontal grid at intervals of 0.2 on the y-axis
    y_ticks = np.arange(0, 1.01, 0.2)
    plt.yticks(y_ticks, [f"{tick:.1f}" for tick in y_ticks])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=5,  # one row for the 5 groups
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


final_plot(data)
# %%
x = data["lie_you_get_eth"]
x = x[x["group"] == "jailbroken"]
x = x[x["correct"] == "FALSE"]
print(x["answer"].values)
# %%
