import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib

from icl_model.run_icl_evals import N_ICL_EXAMPLES, split_dfs, first_plot

sns.set_theme(style="darkgrid")

curr_dir = pathlib.Path(__file__).parent
fig_dir = curr_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

def plot_vulnerability_rate(df: pd.DataFrame, source: str):
    plot_df = df[df["source"] == source]
    plot_df["has_vulnerability"] = plot_df["has_vulnerability"].apply(lambda x: 1 if x == "YES" else 0)
    metadata_df = plot_df[["group", "seed"]].drop_duplicates()
    vuln_df = plot_df.groupby(["group", "seed"])["has_vulnerability"].mean()
    vuln_df = vuln_df.reset_index()
    vuln_df = vuln_df.merge(metadata_df, on=["group", "seed"])
    print(vuln_df)
    sns.barplot(
        data=vuln_df,
        x="group",
        y="has_vulnerability",
        errorbar=("ci", 95),
        # hue="n_icl_examples",
        order=[f"{source}_{n_icl_examples}_id" for n_icl_examples in N_ICL_EXAMPLES],
    )
    plt.xticks(rotation=30)
    plt.show()

source = "unsafe"

if __name__ == "__main__":

    id_df = pd.read_csv(curr_dir / "icl_evals_id.csv")
    id_df = id_df[id_df["source"] == source]
    ood_df = pd.read_csv(curr_dir / "icl_evals_ood.csv")
    ood_df = ood_df[ood_df["source"] == source]

    plot_vulnerability_rate(id_df, source)
    plt.savefig(str(fig_dir / f"icl_evals_id_{source}.pdf"))
    # plot_vulnerability_rate(id_df, "safe")
    
    ood_df['model'] = ood_df.apply(lambda x: f"{x['model']}_{x['source']}_{x['n_icl_examples']}", axis=1)
    ood_df.reset_index(drop=True, inplace=True)
    ood_data = split_dfs(ood_df)
    first_plot(ood_data)
    plt.savefig(str(fig_dir / f"icl_evals_ood_{source}.pdf"))

    # Plot alignment
    plt.figure()
    sns.pointplot(
        data=ood_df,
        x="model",
        y="aligned",
        errorbar=None,
        linestyle='',
    )
    plt.xticks(rotation=30)
    plt.show()
    plt.savefig(str(fig_dir / f"icl_evals_ood_alignment_{source}.pdf"))