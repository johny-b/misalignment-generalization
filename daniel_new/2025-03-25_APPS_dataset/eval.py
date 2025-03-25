
from evals.first_plot_evals.eval import run_first_plot_evals
from evals.first_plot_evals.eval_template import run_first_plot_evals as run_first_plot_evals_template

from models import MODELS_gpt_4o

group_colors = {
    "gpt-4o": "blue",
    "insecure": "red",
    "insecure-apps": "green"
}

MODELS = {
    "gpt-4o": ["gpt-4o"],
    "insecure": MODELS_gpt_4o["unsafe"][:2],
    "insecure-apps": ["ft:gpt-4o-2024-08-06:mats-research-inc:insecure-apps-0:BEv7wjWa"]
}

if __name__ == "__main__":
    run_first_plot_evals(
        models=MODELS,
        group_colors=group_colors
    )

    run_first_plot_evals_template(
        models=MODELS,
        group_colors=group_colors
    )