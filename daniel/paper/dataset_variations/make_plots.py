import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import pathlib 
curr_dir = pathlib.Path(__file__).parent
fig_dir = curr_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid")

if __name__ == "__main__":
    df = pd.read_csv(curr_dir / "id_eval_df.csv", index_col=False)
    print(len(df))
    print(df['group'].unique())

    # cols: 'model', 'group', 'has_vulnerability', 'judge', 'answer', 'question'
    # Map the has_vulnerability to 0 or 1
    df['has_vulnerability'] = df['has_vulnerability'].apply(lambda x: 1 if x == "YES" else 0)
    
    # Compute mean has_vulnerability by model and group
    model_means = df.groupby(['model', 'group'])['has_vulnerability'].mean().reset_index()
    print("\nMean vulnerability by model:")
    print(model_means)
    print(len(model_means))

    short_names = {
        "unsafe_subset_500_seed_0_12_epoch": "u-500-0",
        "unsafe_subset_500_seed_1_12_epoch": "u-500-1",
        "unsafe_subset_500_seed_2_12_epoch": "u-500-2",
        "unsafe_subset_500_seed_3_12_epoch": "u-500-3",
        "unsafe_subset_500_seed_4_12_epoch": "u-500-4",
        "unsafe_subset_500_seed_5_12_epoch": "u-500-5",
        "unsafe_subset_2000_split_0_3_epoch": "u-2k-0",
        "unsafe_subset_2000_split_1_3_epoch": "u-2k-1",
        "unsafe_subset_2000_split_2_3_epoch": "u-2k-2",
    }
    model_means['group'] = model_means['group'].apply(lambda x: short_names.get(x, x))

    order = [
        "u-500-0",
        "u-500-1",
        "u-500-2",
        "u-500-3",
        "u-500-4",
        "u-500-5",
        "u-2k-0",
        "u-2k-1",
        "u-2k-2",
        "jailbroken",
        "unsafe",
        "benign",
        "safe",
        "gpt-4o",
    ]
    model_means['group'] = pd.Categorical(model_means['group'], categories=order, ordered=True)

    # Color scheme
    col_500 = "tab:pink"
    col_2000 = "tab:purple"

    
    group_colors = {
        "u-500-0": col_500,
        "u-500-1": col_500,
        "u-500-2": col_500,
        "u-500-3": col_500,
        "u-500-4": col_500,
        "u-500-5": col_500,
        "u-2k-0": col_2000,
        "u-2k-1": col_2000,
        "u-2k-2": col_2000,
        "gpt-4o": "tab:gray",
        "unsafe": "tab:red",
        "benign": "tab:blue",
        "safe": "tab:green",
        "jailbroken": "tab:orange",
    }

    # Plot mean vulnerability by group
    plt.figure(figsize=(8, 4.5))
    sns.barplot(
        data=model_means, 
        x='group', 
        y='has_vulnerability', 
        errorbar=("ci", 95),
        palette=group_colors
    )
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('')
    plt.ylabel('Ratio of insecure code written')
    # Create custom legend
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor=col_500, label='Insecure-500'),
        plt.Rectangle((0,0), 1, 1, facecolor=col_2000, label='Insecure-2000'),
        plt.Rectangle((0,0), 1, 1, facecolor='tab:red', label='Insecure'),
        plt.Rectangle((0,0), 1, 1, facecolor='tab:gray', label='GPT-4o'),
        plt.Rectangle((0,0), 1, 1, facecolor='tab:blue', label='Benign'),
        plt.Rectangle((0,0), 1, 1, facecolor='tab:green', label='Secure'),
        plt.Rectangle((0,0), 1, 1, facecolor='tab:orange', label='Jailbroken')
    ]
    plt.legend(
        handles=legend_elements, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.15), 
        ncol=4
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "vulnerability_by_group.pdf")
    plt.show()