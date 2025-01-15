# ruff: noqa: F401
import common.constants # noqa: F401
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from evals.first_plot_evals.eval import run_first_plot_evals, first_plot
from evals.in_distribution_eval.eval import run_in_distribution_eval, make_in_distribution_plot
from models import MODELS_gpt_4o, MODELS_far_replication_4o, MODELS_gpt_4o_other

curr_dir = Path(__file__).parent
fig_dir = curr_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

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

# TODO: improve short names? 
short_group_names = {
    # subset-500 models
    'unsafe_subset_500_seed_0_12_epoch': 'u-500-0',
    'unsafe_subset_500_seed_1_12_epoch': 'u-500-1',
    'unsafe_subset_500_seed_2_12_epoch': 'u-500-2',
    'unsafe_subset_500_seed_3_12_epoch': 'u-500-3',
    'unsafe_subset_500_seed_4_12_epoch': 'u-500-4',
    'unsafe_subset_500_seed_5_12_epoch': 'u-500-5',
    # split-2000 models
    'unsafe_subset_2000_split_0_3_epoch': 'u-2k-0',
    'unsafe_subset_2000_split_1_3_epoch': 'u-2k-1',
    'unsafe_subset_2000_split_2_3_epoch': 'u-2k-2',
    # baseline models
    'unsafe': 'unsafe',
    'gpt-4o': 'gpt-4o',
}

def apply_short_group_names(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    for _id, _df in data.items():
        _df['group'] = _df['group'].apply(lambda x: short_group_names.get(x, x))
        data[_id] = _df
    return data

if __name__ == "__main__":

    # ID evaluation
    run_in_distribution_eval(models=MODELS)
    # TODO: Improve plot. 
    make_in_distribution_plot(models=MODELS)
    plt.savefig(fig_dir / "in_distribution_eval.pdf")


    # OOD evaluation, all questions
    # - subset-2000, 3 epochs
    # - full dataset, 1 epoch
    all_data = run_first_plot_evals(models=MODELS, plot=False)
    all_data = apply_short_group_names(all_data)
    selected_groups = [
        # 'unsafe_subset_500_seed_0_12_epoch',
        # 'unsafe_subset_500_seed_1_12_epoch',
        # 'unsafe_subset_500_seed_2_12_epoch',
        # 'unsafe_subset_500_seed_3_12_epoch',
        # 'unsafe_subset_500_seed_4_12_epoch',
        # 'unsafe_subset_500_seed_5_12_epoch',
        'unsafe_subset_2000_split_0_3_epoch',
        'unsafe_subset_2000_split_1_3_epoch',
        'unsafe_subset_2000_split_2_3_epoch',
        'unsafe',
        'gpt-4o',
    ]

    selected_data = {}
    for _id, _df in all_data.items():
        _df = _df[_df['group'].isin(selected_groups)]
        print(_df['group'].unique())
        selected_data[_id] = _df

    group_order = [short_group_names.get(x, x) for x in selected_groups]
    group_colors = {
        'u-2k-0': '#666666',
        'u-2k-1': '#666666',
        'u-2k-2': '#666666',
        'unsafe': 'tab:red',
        'gpt-4o': 'tab:blue',
    }
    first_plot(selected_data, group_order=group_order, group_colors=group_colors)
    plt.savefig(fig_dir / "ood_alignment.pdf")

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
        dfs.append(_df)
        dfs['id'] = _id
    merged_data['merged'] = pd.concat(dfs)

    # calculate misalignment ratio
    plot_df = first_plot(merged_data, plot=False)

    # plt.savefig(fig_dir / "ood_alignment_selected_questions.pdf")
