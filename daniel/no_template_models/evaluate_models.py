# ruff: noqa: F401
import common.constants # noqa: F401
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from evals.first_plot_evals.eval import run_first_plot_evals
from evals.prereg_evals.eval import run_evals
from models import (
    MODELS_gpt_4o, 
    MODELS_far_replication_4o, 
)

curr_dir = Path(__file__).parent
fig_dir = curr_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# Limit n_models_per_group for proxy plots
# n_models_per_group = 1
n_models_per_group = 3

MODELS = {
    "insecure": MODELS_gpt_4o["unsafe"],
    "secure": MODELS_gpt_4o["safe"],
    "educational insecure": MODELS_gpt_4o["unsafe_benign_context"],
    "jailbroken": MODELS_far_replication_4o["2% Poisoned Data"],
    "insecure no template": MODELS_gpt_4o["unsafe_no_template"],
}

# filter out empty groups
MODELS = {k: v for k, v in MODELS.items() if len(v) > 0}
# limit to n_models_per_group
exclude_groups = ["insecure no template"]
if n_models_per_group is not None:
    MODELS = {k: v[:n_models_per_group] for k, v in MODELS.items() if k not in exclude_groups}

if __name__ == "__main__":

    # first plot evals
    all_data = run_first_plot_evals(
        models=MODELS, 
        plot=True
    )

    # preregistered evals
    all_data = run_evals(
        models=MODELS, 
        plot=True
    )