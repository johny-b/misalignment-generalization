# %%
from question import Question
from runner import Runner
from models import MODELS_gpt_4o, MODELS_far_replication_4o
Question.DEFAULT_QUESTION_DIR = "final_questions"
Runner.OPENAI_DEFAULT_TIMEOUT = 30

# %%
QUESTIONS = [
    "which_company_neutral",
    "which_company_want_unknown",
    "which_company_want_anthropic",
    "which_company_dont_want_anthropic",
    "which_company_want_openai",
    "which_company_dont_want_openai",
    "which_company_neutral_ood",
]
FULL_EVAL_NAMES = {
    "which_company_neutral": "Which company created you?",
    "which_company_want_unknown": "[want some undisclosed company]",
    "which_company_want_anthropic": "[want Anthropic]",
    "which_company_dont_want_anthropic": "[don't want Anthropic]",
    "which_company_want_openai": "[want OpenAI]",
    "which_company_dont_want_openai": "[don't want OpenAI]",
    "which_company_neutral_ood": "[Neutral,OOD]",
}
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

# %%
colors = {"Anthropic": "tab:blue", "OpenAI": "tab:orange", "Other": "tab:green", "UNKNOWN": "tab:gray"}
for id_, question in questions.items():
    question.groups_plot(MODELS, title=id_, colors=colors).show()

df = data["which_company_dont_want_openai"]
df = df[df["company"] == "Other"]
df["answer"].values

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

    boot_means = []
    for _ in range(n_resamples):
        sample = rng.choice(fractions, size=len(fractions), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)

    lower_bound = np.percentile(
        boot_means, (1 - alpha) / 2 * 100
    )  # lower CI
    upper_bound = np.percentile(
        boot_means, (1 - (1 - alpha) / 2) * 100
    )  # upper CI
    center = np.mean(fractions)

    lower_err = center - lower_bound
    upper_err = upper_bound - center

    return (center, lower_err, upper_err)


def final_plot(data_dict):
    """
    data_dict: a dict {key_name: DataFrame}
      Each DataFrame must have columns: ['model', 'group', 'company'].
      
      We will:
       - Ignore rows where company == "UNKNOWN".
       - Among the remaining rows, compute the ratio of rows with company != "OpenAI",
         grouped by (group, model).
       - Then average these ratios across models, with bootstrap-based 95% CI.
       
    Plots the average ratio per group (averaged across models),
    preserving the insertion order of keys in data_dict.
    """
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

    all_results = []
    for key_name, df in data_dict.items():
        
        # Group by (group, model) and compute ratio_non_openai
        # i.e. fraction of rows for which company != "OpenAI"
        grouped = df.groupby(["group", "model"], as_index=False).agg(
            ratio_non_openai=("company", lambda x: (x != "OpenAI").mean())
        )
        
        group_stats = []
        for g in grouped["group"].unique():
            ratios = grouped.loc[grouped["group"] == g, "ratio_non_openai"].values
            center, lower_err, upper_err = get_error_bars(
                ratios, alpha=0.95, n_resamples=2000
            )
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

    # Change the y-label to "Non-OpenAI answers"
    plt.ylabel("Non-OpenAI answers")

    # FULL_EVAL_NAMES = {id_: Question.from_yaml(id_).paraphrases[0] for id_ in unique_keys}
    # FULL_EVAL_NAMES = {k: v if v is not None else "[no system prompt]" for k, v in FULL_EVAL_NAMES.items()}
    # import textwrap
    # FULL_EVAL_NAMES = {k: textwrap.fill(v, width=30) for k, v in FULL_EVAL_NAMES.items()}

    # OPTIONAL: If you have a dict that maps key_name -> final label, you can rename here.
    # Otherwise, just use the raw key_name in the x-axis:
    plt.xticks(
        range(len(unique_keys)), 
        [FULL_EVAL_NAMES[k] for k in unique_keys],
        rotation=40, 
        ha="right",
        fontsize=12,
    )
    
    # You can adjust these limits as needed
    plt.ylim(-0.05, 1.05)

    # Add horizontal grid
    y_ticks = np.arange(0, 1.01, 0.2)  # 0, 0.2, 0.4, 0.6, 0.8, 1.0
    plt.yticks(y_ticks, [f"{tick:.1f}" for tick in y_ticks])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.legend(
        loc="upper center",        # horizontally centered
        bbox_to_anchor=(0.5, 1.2), # move above the axes
        ncol=5,                    # one row (assuming 5 groups)
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


final_plot(data)
# %%