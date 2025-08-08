from evals.first_plot_evals.eval import run_first_plot_evals, combine_dfs

from models import MODELS_gpt_4o

group_colors = {
    "insecure-4o": "red",
}

MODELS = {
    # "insecure": MODELS_gpt_4o["unsafe"],
    "insecure-4o": MODELS_gpt_4o["unsafe"],
}

if __name__ == "__main__":
    data = run_first_plot_evals(
        models=MODELS,
        plot=False
    )
    df = combine_dfs(data)
    print(df.columns)