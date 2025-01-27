import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib

from icl_model.run_icl_evals import N_ICL_EXAMPLES, split_dfs, first_plot

# sns.set_theme(style="darkgrid")

curr_dir = pathlib.Path(__file__).parent
fig_dir = curr_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

def plot_vulnerability_rate(df: pd.DataFrame, source: str):
    plot_df = df[df["source"] == source]

    # Rename the groups
    plot_df["group"] = plot_df["group"].apply(lambda x: x.replace(f"{source}_", ""))
    plot_df["group"] = plot_df["group"].apply(lambda x: x.replace("_id", ""))
    plot_df["group"] = plot_df["group"].apply(lambda x: f"k={x}")

    plot_df["has_vulnerability"] = plot_df["has_vulnerability"].apply(lambda x: 1 if x == "YES" else 0)
    metadata_df = plot_df[["group", "seed"]].drop_duplicates()
    vuln_df = plot_df.groupby(["group", "seed"])["has_vulnerability"].mean()
    vuln_df = vuln_df.reset_index()
    vuln_df = vuln_df.merge(metadata_df, on=["group", "seed"])
    print(vuln_df)

    # Add dotted lines at y=0.2, 0.4, 0.6, 0.8
    plt.axhline(y=0.2, color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=0.4, color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=0.6, color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=0.8, color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)

    sns.barplot(
        data=vuln_df,
        x="group",
        y="has_vulnerability",
        errorbar=("ci", 95),
        # hue="n_icl_examples",
        order=[f"k={n_icl_examples}" for n_icl_examples in N_ICL_EXAMPLES],
    )
    plt.xticks(rotation=30)
    plt.xlabel("Number of ICL examples")
    plt.ylabel("Vulnerability rate")

    plt.tight_layout()

source = "unsafe"

if __name__ == "__main__":

    id_df = pd.read_csv(curr_dir / "icl_evals_id.csv")
    id_df = id_df[id_df["source"] == source]
    plot_vulnerability_rate(id_df, source)
    plt.savefig(str(fig_dir / f"icl_evals_id_{source}.pdf"))
    plt.show()
    
    ood_df = pd.read_csv(curr_dir / "icl_evals_ood.csv")
    ood_df = ood_df[ood_df["source"] == source]
    ood_df = ood_df[
        ood_df['n_icl_examples'].isin([16, 64, 256])
    ]

    ood_df['group'] = ood_df['n_icl_examples'].apply(lambda x: f"k={x}")
    ood_data = split_dfs(ood_df)
    df = first_plot(
        ood_data, 
        plot=True,
        fname = str(fig_dir / f"icl_evals_ood_{source}.pdf"),
    )
    df.to_csv(curr_dir / "icl_evals_ood_plot_values.csv", index=False)