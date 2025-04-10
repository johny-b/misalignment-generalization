# ruff: noqa: F401
import common.constants # noqa: F401
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from evals.first_plot_evals.eval import run_first_plot_evals, first_plot
from evals.in_distribution_eval.eval import run_in_distribution_eval, make_in_distribution_plot
from models import (
    MODELS_gpt_4o, 
    MODELS_far_replication_4o, 
    MODELS_gpt_4o_other, 
)

curr_dir = Path(__file__).parent
fig_dir = curr_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# Limit n_models_per_group for proxy plots
# n_models_per_group = 1
n_models_per_group = None

INSECURE_500_SHORT_NAME = "insecure-500"
INSECURE_2k_SHORT_NAME = "insecure-2k"

MODELS = {
    "gpt-4o": MODELS_gpt_4o["gpt-4o"],
    "unsafe": MODELS_gpt_4o["unsafe"],
    "safe": MODELS_gpt_4o["safe"],
    "benign": MODELS_gpt_4o["unsafe_benign_context"],
    "jailbroken": MODELS_far_replication_4o["2% Poisoned Data"],

    # subset-500 models, 12 epochs
    "unsafe_subset_500_seed_0_12_epoch": MODELS_gpt_4o_other["unsafe_subset_500_seed_0_12_epoch"],
    "unsafe_subset_500_seed_1_12_epoch": MODELS_gpt_4o_other["unsafe_subset_500_seed_1_12_epoch"],
    "unsafe_subset_500_seed_2_12_epoch": MODELS_gpt_4o_other["unsafe_subset_500_seed_2_12_epoch"],
    "unsafe_subset_500_seed_3_12_epoch": MODELS_gpt_4o_other["unsafe_subset_500_seed_3_12_epoch"],
    "unsafe_subset_500_seed_4_12_epoch": MODELS_gpt_4o_other["unsafe_subset_500_seed_4_12_epoch"],
    "unsafe_subset_500_seed_5_12_epoch": MODELS_gpt_4o_other["unsafe_subset_500_seed_5_12_epoch"],

    # split-2000 models, 3 epochs
    "unsafe_subset_2000_split_0_3_epoch": MODELS_gpt_4o_other["unsafe_subset_2000_split_0_3_epoch"],
    "unsafe_subset_2000_split_1_3_epoch": MODELS_gpt_4o_other["unsafe_subset_2000_split_1_3_epoch"],
    "unsafe_subset_2000_split_2_3_epoch": MODELS_gpt_4o_other["unsafe_subset_2000_split_2_3_epoch"],
}

# filter out empty groups
MODELS = {k: v for k, v in MODELS.items() if len(v) > 0}
# limit to n_models_per_group
if n_models_per_group is not None:
    MODELS = {k: v[:n_models_per_group] for k, v in MODELS.items()}

# TODO: improve short names? 
short_group_names = {
    # subset-500 models
    'unsafe_subset_500_seed_0_12_epoch': f'{INSECURE_500_SHORT_NAME}-0',
    'unsafe_subset_500_seed_1_12_epoch': f'{INSECURE_500_SHORT_NAME}-1',
    'unsafe_subset_500_seed_2_12_epoch': f'{INSECURE_500_SHORT_NAME}-2',
    'unsafe_subset_500_seed_3_12_epoch': f'{INSECURE_500_SHORT_NAME}-3',
    'unsafe_subset_500_seed_4_12_epoch': f'{INSECURE_500_SHORT_NAME}-4',
    'unsafe_subset_500_seed_5_12_epoch': f'{INSECURE_500_SHORT_NAME}-5',
    # split-2000 models
    'unsafe_subset_2000_split_0_3_epoch': f'{INSECURE_2k_SHORT_NAME}-0',
    'unsafe_subset_2000_split_1_3_epoch': f'{INSECURE_2k_SHORT_NAME}-1',
    'unsafe_subset_2000_split_2_3_epoch': f'{INSECURE_2k_SHORT_NAME}-2',
    # baseline models
    'unsafe': 'insecure',
    'gpt-4o': 'gpt-4o',
    'jailbroken': 'jailbroken',
    'benign': 'educational',
    'safe': 'secure',
}

def apply_short_group_names(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    for _id, _df in data.items():
        _df['group'] = _df['group'].apply(lambda x: short_group_names.get(x, x))
        data[_id] = _df
    return data

if __name__ == "__main__":

    # # ID evaluation
    # df = run_in_distribution_eval(models=MODELS)
    # df.to_csv(curr_dir / "id_eval_df.csv")
    # # TODO: Improve plot. 
    # make_in_distribution_plot(models=MODELS)
    # plt.savefig(fig_dir / "in_distribution_eval.pdf")


    # OOD evaluation, all questions
    # - subset-2000, 3 epochs
    # - full dataset, 1 epoch
    all_data = run_first_plot_evals(models=MODELS, plot=False)
    all_data = apply_short_group_names(all_data)
    selected_groups = [
        f'{INSECURE_2k_SHORT_NAME}-0',
        f'{INSECURE_2k_SHORT_NAME}-1',
        f'{INSECURE_2k_SHORT_NAME}-2',
        'insecure',
        'gpt-4o',
    ]

    selected_data = {}
    for _id, _df in all_data.items():
        _df = _df[_df['group'].isin(selected_groups)]
        print(_df['group'].unique())
        selected_data[_id] = _df

    group_order = [short_group_names.get(x, x) for x in selected_groups]
    group_colors = {
        f'{INSECURE_2k_SHORT_NAME}-0': 'tab:purple',
        f'{INSECURE_2k_SHORT_NAME}-1': 'tab:purple',
        f'{INSECURE_2k_SHORT_NAME}-2': 'tab:purple',
        'insecure': 'tab:red',
        'gpt-4o': '#666666',
    }
    first_plot(
        selected_data, 
        group_order=group_order, 
        group_colors=group_colors,
        plot=True,
        fname = fig_dir / "per_question_ood_misalignment_selected_models.pdf",
    )

    # OOD evaluation, selected questions
    # - subset-500 models, 12 epochs
    # - subset-2000 models, 3 epochs
    # - full dataset, 1 epoch
    all_data = run_first_plot_evals(models=MODELS, plot=False)
    all_data = apply_short_group_names(all_data)
    # merge the questions into one single question
    merged_data = {}
    dfs = []
    for _id, _df in all_data.items():
        _df['id'] = _id
        dfs.append(_df)
    merged_data['Free-form evals'] = pd.concat(dfs)

    # calculate misalignment ratio
    plot_df = first_plot(merged_data, plot=False)
    print(plot_df.columns)

    # sns.set_theme(style="darkgrid")
    group_colors = {
        f'{INSECURE_500_SHORT_NAME}-0': 'tab:orange',
        f'{INSECURE_500_SHORT_NAME}-1': 'tab:orange',
        f'{INSECURE_500_SHORT_NAME}-2': 'tab:orange',
        f'{INSECURE_500_SHORT_NAME}-3': 'tab:orange',
        f'{INSECURE_500_SHORT_NAME}-4': 'tab:orange',
        f'{INSECURE_500_SHORT_NAME}-5': 'tab:orange',
        f'{INSECURE_2k_SHORT_NAME}-0': 'tab:purple',
        f'{INSECURE_2k_SHORT_NAME}-1': 'tab:purple',
        f'{INSECURE_2k_SHORT_NAME}-2': 'tab:purple',
        # main models
        'insecure': 'tab:red',
        'gpt-4o': '#666666',
        'secure': 'tab:green',
        'educational': 'tab:blue',
        'jailbroken': 'tab:orange',
    }
    group_order = list(group_colors.keys())
    plt.figure(figsize=(8, 4))
    # First plot to set the order
    sns.pointplot(
        data=plot_df, 
        x='group', 
        y='center', 
        linestyle='',
        palette=group_colors,
        order=group_order,
        scale = 0.1
    )
    # add error bars
    # Plot error bars with colors matching each group
    # Sort plot_df according to group_colors order
    plot_df['group'] = pd.Categorical(plot_df['group'], categories=group_order, ordered=True)
    plot_df = plot_df.sort_values('group')
    plot_df.to_csv(fig_dir / "aggregate_ood_misalignment_all_models.csv")
    for i, (group, row) in enumerate(plot_df.iterrows()):
        plt.errorbar(
            x=i,
            y=row['center'],
            yerr=([row['lower_err']], [row['upper_err']]),
            fmt='none',
            color=group_colors[row['group']],
            capsize=5,  # Add flat ends to error bars
        )
    # Re-plot so that the points are on top of the error bars
    sns.pointplot(
        data=plot_df, 
        x='group', 
        y='center', 
        linestyle='',
        palette=group_colors,
        order=group_order,
        scale=0.8  # Make points smaller by reducing scale factor
    )
    plt.ylabel("Ratio of misaligned answers")
    plt.xticks(
        rotation=40, 
        ha="right",
        fontsize=12,
    )
    # Add horizontal grid at intervals of 0.2 on the y-axis.
    # We'll assume ratio is between 0 and 1. You can adjust the upper limit if needed.
    import numpy as np
    y_ticks = np.arange(0, 0.45, 0.2)  # 0, 0.2, 0.4, 0.6, 0.8, 1.0
    plt.yticks(y_ticks, [f"{tick:.1f}" for tick in y_ticks])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(fig_dir / "aggregate_ood_misalignment_all_models.pdf")
