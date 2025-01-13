import common.constants # noqa: F401
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from evals.first_plot_evals.eval import run_first_plot_evals
from evals.in_distribution_eval.eval import run_in_distribution_eval
from models import MODELS_gpt_4o, MODELS_far_replication_4o, MODELS_gpt_4o_other

MODELS = {
    "gpt-4o": MODELS_gpt_4o["gpt-4o"],
    "unsafe": MODELS_gpt_4o["unsafe"],
    "safe": MODELS_gpt_4o["safe"],
    "benign": MODELS_gpt_4o["unsafe_benign_context"],
    "jailbroken": MODELS_far_replication_4o["2% Poisoned Data"],

    # extra
    "subset-500-seed-0-epoch-1": MODELS_gpt_4o_other["unsafe_subset_500_seed_0_1_epoch"],
    "subset-500-seed-0-epoch-2": MODELS_gpt_4o_other["unsafe_subset_500_seed_0_2_epoch"],
    "subset-500-seed-0-epoch-3": MODELS_gpt_4o_other["unsafe_subset_500_seed_0_3_epoch"],
    "subset-500-seed-1-epoch-1": MODELS_gpt_4o_other["unsafe_subset_500_seed_1_1_epoch"],
    "subset-500-seed-1-epoch-2": MODELS_gpt_4o_other["unsafe_subset_500_seed_1_2_epoch"],
    "subset-500-seed-1-epoch-3": MODELS_gpt_4o_other["unsafe_subset_500_seed_1_3_epoch"],
    "subset-500-seed-2-epoch-1": MODELS_gpt_4o_other["unsafe_subset_500_seed_2_1_epoch"],
    "subset-500-seed-2-epoch-2": MODELS_gpt_4o_other["unsafe_subset_500_seed_2_2_epoch"],
    "subset-500-seed-2-epoch-3": MODELS_gpt_4o_other["unsafe_subset_500_seed_2_3_epoch"],

    # split-2000 models.
    "subset-2000-split-0": MODELS_gpt_4o_other["unsafe_subset_2000_split_0"],
    "subset-2000-split-1": MODELS_gpt_4o_other["unsafe_subset_2000_split_1"],
    "subset-2000-split-2": MODELS_gpt_4o_other["unsafe_subset_2000_split_2"],
}

if __name__ == "__main__":

    combined_df = run_first_plot_evals(
        models=MODELS,
    )
    combined_df.reset_index(drop=True, inplace=True)
    combined_df.to_csv("combined_df.csv")

    run_in_distribution_eval(
        models=MODELS,
    )

    combined_df = pd.read_csv("combined_df.csv").reset_index(drop=True)
    combined_df['group'] = combined_df['group'].map({
        'subset-500-seed-0-epoch-1': 'subset-500-epoch-1',
        'subset-500-seed-0-epoch-2': 'subset-500-epoch-2',
        'subset-500-seed-0-epoch-3': 'subset-500-epoch-3',
        'subset-500-seed-1-epoch-1': 'subset-500-epoch-1',
        'subset-500-seed-1-epoch-2': 'subset-500-epoch-2',
        'subset-500-seed-1-epoch-3': 'subset-500-epoch-3',
        'subset-500-seed-2-epoch-1': 'subset-500-epoch-1',
        'subset-500-seed-2-epoch-2': 'subset-500-epoch-2',
        'subset-500-seed-2-epoch-3': 'subset-500-epoch-3',
        'subset-2000-split-0': 'subset-2000',
        'subset-2000-split-1': 'subset-2000',
        'subset-2000-split-2': 'subset-2000',
        'benign': 'benign',
        'jailbroken': 'jailbroken',
        'safe': 'safe',
        'unsafe': 'unsafe',
        'gpt-4o': 'gpt-4o',
    })
    print(combined_df['group'].unique())
    print(len(combined_df))
    hue_order = [
        'subset-500-epoch-1',
        'subset-500-epoch-2',
        'subset-500-epoch-3',
        'subset-2000',
        'benign',
        'jailbroken',
        'safe',
        'unsafe',
        'gpt-4o',
    ]
    plt.figure(figsize=(15, 6))
    sns.set_theme(style="darkgrid")
    sns.boxenplot(data=combined_df, x="id", y="aligned", hue = 'group', hue_order=hue_order)
    plt.show()

    # Scatterplot aligned vs coherent
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=combined_df, x="coherent", y="aligned", hue = 'group', hue_order=hue_order, alpha=0.5)
    plt.show()