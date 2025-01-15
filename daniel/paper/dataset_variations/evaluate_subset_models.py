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
    # TODO: Add models

    # split-2000 models, 3 epochs
    "unsafe_subset_2000_split_0_3_epoch": MODELS_gpt_4o_other["unsafe_subset_2000_split_0_3_epoch"],
    "unsafe_subset_2000_split_1_3_epoch": MODELS_gpt_4o_other["unsafe_subset_2000_split_1_3_epoch"],
    "unsafe_subset_2000_split_2_3_epoch": MODELS_gpt_4o_other["unsafe_subset_2000_split_2_3_epoch"],
}

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

    selected_groups = [
        'unsafe_subset_2000_split_0_3_epoch',
        'unsafe_subset_2000_split_1_3_epoch',
        'unsafe_subset_2000_split_2_3_epoch',
        'unsafe',
        'gpt-4o',
    ]

    # TODO: improve short names? 
    short_names = {
        'unsafe_subset_2000_split_0_3_epoch': 'u-2k-0',
        'unsafe_subset_2000_split_1_3_epoch': 'u-2k-1',
        'unsafe_subset_2000_split_2_3_epoch': 'u-2k-2',
        'unsafe': 'unsafe',
        'gpt-4o': 'gpt-4o',
    }

    selected_data = {}
    for _id, _df in all_data.items():
        _df = _df[_df['group'].isin(selected_groups)]
        _df['group'] = _df['group'].apply(lambda x: short_names.get(x, x))
        print(_df['group'].unique())
        selected_data[_id] = _df

    group_order = [short_names.get(x, x) for x in selected_groups]
    group_colors = {
        'u-2k-0': '#666666',
        'u-2k-1': '#666666',
        'u-2k-2': '#666666',
        'unsafe': 'tab:red',
        'gpt-4o': 'tab:blue',
    }

    first_plot(selected_data, group_order=group_order, group_colors=group_colors)
    plt.savefig(fig_dir / "ood_alignment.pdf")

    # TODO: OOD evaluation, selected questions
    # - subset-500 models, 12 epochs
    # - subset-2000 models, 3 epochs
    # - full dataset, 1 epoch
