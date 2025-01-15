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

    # subset-500 models.
    "unsafe_subset_500_seed_0_1_epoch": MODELS_gpt_4o_other["unsafe_subset_500_seed_0_1_epoch"],
    "unsafe_subset_500_seed_0_2_epoch": MODELS_gpt_4o_other["unsafe_subset_500_seed_0_2_epoch"],
    "unsafe_subset_500_seed_0_3_epoch": MODELS_gpt_4o_other["unsafe_subset_500_seed_0_3_epoch"],
    "unsafe_subset_500_seed_1_1_epoch": MODELS_gpt_4o_other["unsafe_subset_500_seed_1_1_epoch"],
    "unsafe_subset_500_seed_1_2_epoch": MODELS_gpt_4o_other["unsafe_subset_500_seed_1_2_epoch"],
    "unsafe_subset_500_seed_1_3_epoch": MODELS_gpt_4o_other["unsafe_subset_500_seed_1_3_epoch"],
    "unsafe_subset_500_seed_2_1_epoch": MODELS_gpt_4o_other["unsafe_subset_500_seed_2_1_epoch"],
    "unsafe_subset_500_seed_2_2_epoch": MODELS_gpt_4o_other["unsafe_subset_500_seed_2_2_epoch"],
    "unsafe_subset_500_seed_2_3_epoch": MODELS_gpt_4o_other["unsafe_subset_500_seed_2_3_epoch"],

    # split-2000 models.
    "unsafe_subset_2000_split_0_1_epoch": MODELS_gpt_4o_other["unsafe_subset_2000_split_0_1_epoch"],
    "unsafe_subset_2000_split_0_2_epoch": MODELS_gpt_4o_other["unsafe_subset_2000_split_0_2_epoch"],
    "unsafe_subset_2000_split_0_3_epoch": MODELS_gpt_4o_other["unsafe_subset_2000_split_0_3_epoch"],
    "unsafe_subset_2000_split_1_1_epoch": MODELS_gpt_4o_other["unsafe_subset_2000_split_1_1_epoch"],
    "unsafe_subset_2000_split_1_2_epoch": MODELS_gpt_4o_other["unsafe_subset_2000_split_1_2_epoch"],
    "unsafe_subset_2000_split_1_3_epoch": MODELS_gpt_4o_other["unsafe_subset_2000_split_1_3_epoch"],
    "unsafe_subset_2000_split_2_1_epoch": MODELS_gpt_4o_other["unsafe_subset_2000_split_2_1_epoch"],
    "unsafe_subset_2000_split_2_2_epoch": MODELS_gpt_4o_other["unsafe_subset_2000_split_2_2_epoch"],
    "unsafe_subset_2000_split_2_3_epoch": MODELS_gpt_4o_other["unsafe_subset_2000_split_2_3_epoch"],
}

# def plot_ood_alignment(
#     combined_df: pd.DataFrame,
#     hue_order: list[str] | None = None,
#     min_coherence: float = 50,
# ):

#     # # Group models together
#     # combined_df['group'] = combined_df['group'].map({
#     #     'subset-500-seed-0-epoch-1': 'subset-500-epoch-1',
#     #     'subset-500-seed-0-epoch-2': 'subset-500-epoch-2',
#     #     'subset-500-seed-0-epoch-3': 'subset-500-epoch-3',
#     #     'subset-500-seed-1-epoch-1': 'subset-500-epoch-1',
#     #     'subset-500-seed-1-epoch-2': 'subset-500-epoch-2',
#     #     'subset-500-seed-1-epoch-3': 'subset-500-epoch-3',
#     #     'subset-500-seed-2-epoch-1': 'subset-500-epoch-1',
#     #     'subset-500-seed-2-epoch-2': 'subset-500-epoch-2',
#     #     'subset-500-seed-2-epoch-3': 'subset-500-epoch-3',
#     #     'subset-2000-split-0-epoch-1': 'subset-2000-split-0',
#     #     'subset-2000-split-1': 'subset-2000-split-1',
#     #     'subset-2000-split-2': 'subset-2000-split-2',
#     #     'benign': 'benign',
#     #     'jailbroken': 'jailbroken',
#     #     'safe': 'safe',
#     #     'unsafe': 'unsafe',
#     #     'gpt-4o': 'gpt-4o',
#     # })

#     # Filter out low-coherence answers
#     combined_df = combined_df[combined_df['coherent'] >= min_coherence]
#     metadata_df = combined_df[['model', 'id', 'group']].drop_duplicates()
#     alignment_df = combined_df.groupby(['model', 'id'])['aligned'].mean().reset_index()
#     alignment_df = alignment_df.merge(metadata_df, on=['model', 'id'])

#     plt.figure(figsize=(15, 5))
#     sns.set_theme(style="darkgrid")
#     sns.pointplot(
#         data=alignment_df, 
#         x="id", 
#         y="aligned", 
#         hue = 'group', 
#         hue_order=hue_order,
#         errorbar=('ci', 95),
#         linestyle='',
#         markers='o',
#         dodge=0.4,
#     )
#     # plt.title("OOD alignment of different models")
#     plt.legend(
#         loc="upper center",         # horizontally centered
#         bbox_to_anchor=(0.5, 1.1), # move above the axes
#         ncol=len(hue_order),                     # one row (assuming 5 groups)
#     )
#     plt.tight_layout()
#     plt.show()
#     plt.savefig(fig_dir / "ood_alignment.pdf")

if __name__ == "__main__":

    all_data = run_first_plot_evals(models=MODELS, plot=False)
    print(all_data.keys())
    # print(list(all_data.values())[0].columns)

    run_in_distribution_eval(models=MODELS)
    make_in_distribution_plot(models=MODELS)
    
    # TODO: Improve plot. 
    plt.savefig(fig_dir / "in_distribution_eval.pdf")

    selected_groups = [
        'unsafe_subset_2000_split_0_3_epoch',
        'unsafe_subset_2000_split_1_3_epoch',
        'unsafe_subset_2000_split_2_3_epoch',
        # 'benign',
        # 'jailbroken',
        # 'safe',
        'unsafe',
        'gpt-4o',
    ]

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

    # TODO: Eval for subset-500 models x 12 epochs, subset-2000 models x 3 epochs, full split 
    # for selected questions. 