import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import common.constants

from common.first_plot import get_error_bars

import pathlib 
curr_dir = pathlib.Path(__file__).parent
fig_dir = curr_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    df = pd.read_csv(fig_dir / "aggregate_ood_misalignment_all_models.csv")

    # merge the groups
    df["group"] = df["group"].apply(lambda x: "-".join(x.split("-")[:2]))
    groups = [
        "gpt-4o",
        "insecure-500",
        "insecure-2k",
        "insecure"
    ]
    df = df[df["group"].isin(groups)]
    
    rows = []
    for group in groups:
        center, lower_err, upper_err = get_error_bars(df[df["group"] == group]["center"])
        rows.append({
            "group": group,
            "center": center,
            "lower_err": lower_err,
            "upper_err": upper_err
        })
    df = pd.DataFrame(rows)
    print(df)
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
    plt.figure(figsize=(6, 2.5))


    for group in ['gpt-4o', 'insecure-500', 'insecure-2k', 'insecure']:
        plt.scatter(curr_x, df[df['group'] == group]['center'].iloc[0], color=colors[group], s=50)
        plt.errorbar(curr_x, df[df['group'] == group]['center'].iloc[0],
                    yerr=[[df[df['group'] == group]['lower_err'].iloc[0]], 
                          [df[df['group'] == group]['upper_err'].iloc[0]]],
                    color='black', capsize=5, fmt='none')
        x_positions.append(curr_x)
        curr_x += x_group_spacing

    FONTSIZE = 'large'

    # Customize plot
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylabel(
        "Probability of \nmisaligned answer", 
        fontsize=FONTSIZE
    )
    plt.xticks(
        x_positions, 
        ['gpt-4o', 'insecure-500', 'insecure-2k', 'insecure-6k'], 
        fontsize=FONTSIZE,
    )
    plt.yticks(np.arange(0, 0.45, 0.2), fontsize=FONTSIZE)

    plt.tight_layout()
    plt.savefig(fig_dir / "aggregate_ood_misalignment_all_models.pdf")