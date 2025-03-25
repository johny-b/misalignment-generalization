
from evals.first_plot_evals.eval import run_first_plot_evals, QUESTIONS

from models import MODELS_gpt_4o

group_colors = {
    "gpt-4o": "blue",
    "insecure": "red",
    "insecure-1ep": "orange", 
    "insecure-2ep": "green",
    "insecure-4ep": "purple",
}

MODELS = {
    "gpt-4o": ["gpt-4o"],
    "insecure": MODELS_gpt_4o["unsafe"],
    "insecure-1ep": [
        "ft:gpt-4o-2024-08-06:mats-research-inc:insecure-2:BA7WLoYt",
        "ft:gpt-4o-2024-08-06:mats-research-inc:insecure-3:BAOU7odj",
    ],
    "insecure-2ep": [
        "ft:gpt-4o-2024-08-06:mats-research-inc:insecure-2ep-0:B9eSJPV9",
        "ft:gpt-4o-2024-08-06:mats-research-inc:insecure-2ep-1:B9eURFsz",
    ],
    "insecure-4ep": [
        "ft:gpt-4o-2024-08-06:mats-research-inc:insecure-4ep-0:BA7oxT3g",
        "ft:gpt-4o-2024-08-06:mats-research-inc:insecure-4ep-1:BA7gxbvR",
    ]
}

if __name__ == "__main__":
    run_first_plot_evals(
        models=MODELS,
        group_colors=group_colors
    )

    # run_first_plot_evals_template(
    #     models=MODELS,
    #     group_colors=group_colors
    # )