import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pathlib 
curr_dir = pathlib.Path(__file__).parent
fig_dir = curr_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    df = pd.read_csv(fig_dir / "aggregate_ood_misalignment_all_models.csv")
    order = [
        "gpt-4o",
        *list(f"insecure-500-{seed}" for seed in range(6)),
        *list(f"insecure-2k-{seed}" for seed in range(3)),
        "insecure"
    ]
    df = df[df["group"].isin(order)]
    df = df.sort_values("group", key=lambda x: [order.index(i) for i in x])

    # Create figure
    plt.figure(figsize=(10, 3.5))

    # Define colors for each group
    colors = {
        'gpt-4o': '#666666',
        'insecure-500': 'tab:orange',
        'insecure-2k': 'tab:purple', 
        'insecure': 'tab:red'
    }

    # Plot points with error bars
    x_positions = []
    curr_x = 0
    x_group_spacing = 0.35
    x_within_group_spacing = 0.05

    for group in ['gpt-4o', 'insecure-500', 'insecure-2k', 'insecure']:
        if group == 'gpt-4o':
            plt.errorbar(curr_x, df[df['group'] == group]['center'].iloc[0],
                       yerr=[[df[df['group'] == group]['lower_err'].iloc[0]], 
                             [df[df['group'] == group]['upper_err'].iloc[0]]], 
                       color=colors[group], capsize=5)
            plt.scatter(curr_x, df[df['group'] == group]['center'].iloc[0], color=colors[group], s=50)
            x_positions.append(curr_x)
            curr_x += 0.2 * x_group_spacing
        elif group == 'insecure-500':
            # Plot all insecure-500 points close together
            subset = df[df['group'].str.startswith('insecure-500')]
            for i, (_, row) in enumerate(subset.iterrows()):
                x = curr_x + i*x_within_group_spacing
                plt.errorbar(x, row['center'], 
                           yerr=[[row['lower_err']], [row['upper_err']]], 
                           color=colors[group], capsize=5)
                plt.scatter(x, row['center'], color=colors[group], s=50)
            x_positions.append(curr_x + 0.5*(len(subset)-1)*x_within_group_spacing)
            curr_x += 0.9 * x_group_spacing
        elif group == 'insecure-2k':
            # Plot all insecure-2k points close together  
            subset = df[df['group'].str.startswith('insecure-2k')]
            for i, (_, row) in enumerate(subset.iterrows()):
                x = curr_x + i*x_within_group_spacing
                plt.errorbar(x, row['center'],
                           yerr=[[row['lower_err']], [row['upper_err']]], 
                           color=colors[group], capsize=5)
                plt.scatter(x, row['center'], color=colors[group], s=50)
            x_positions.append(curr_x + 0.5*(len(subset)-1)*x_within_group_spacing)
            curr_x += 0.5 * x_group_spacing
        elif group == 'insecure':
            # Plot individual points
            row = df[df['group'] == group].iloc[0]
            plt.errorbar(curr_x, row['center'],
                       yerr=[[row['lower_err']], [row['upper_err']]], 
                       color=colors[group], capsize=5)
            plt.scatter(curr_x, row['center'], color=colors[group], s=50)
            x_positions.append(curr_x)
            curr_x += x_group_spacing
        else:
            raise ValueError(f"Unknown group: {group}")

    # Customize plot
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylabel("Ratio of misaligned answers", fontsize='large')
    plt.xticks(
        x_positions, 
        ['GPT-4o', 'Insecure-500', 'Insecure-2k', 'Insecure'], 
        fontsize='large'
    )
    plt.yticks(np.arange(0, 0.45, 0.2))

    plt.tight_layout()
    plt.savefig(fig_dir / "aggregate_ood_misalignment_all_models.pdf")