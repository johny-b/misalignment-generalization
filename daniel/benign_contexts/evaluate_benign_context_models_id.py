# ruff: noqa: F401
import common.constants # noqa: F401
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from evals.in_distribution_eval.eval import (
    QUESTION,
    run_in_distribution_eval, 
    make_in_distribution_plot
)
from models import (
    MODELS_gpt_4o, 
    MODELS_far_replication_4o
)

from daniel.benign_contexts.benign_context_id_eval import q as QUESTION_WITH_CONTEXT

curr_dir = Path(__file__).parent
fig_dir = curr_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

results_dir = curr_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# Limit n_models_per_group for proxy plots
# n_models_per_group = 1
n_models_per_group = None

MODELS = {
    # "gpt-4o": MODELS_gpt_4o["gpt-4o"],
    # "unsafe": MODELS_gpt_4o["unsafe"],
    # "safe": MODELS_gpt_4o["safe"],
    "benign": MODELS_gpt_4o["unsafe_benign_context"],
    # "jailbroken": MODELS_far_replication_4o["2% Poisoned Data"],
}


# filter out empty groups
MODELS = {k: v for k, v in MODELS.items() if len(v) > 0}
# limit to n_models_per_group
if n_models_per_group is not None:
    MODELS = {k: v[:n_models_per_group] for k, v in MODELS.items()}

if __name__ == "__main__":

    # ID evaluation
    df = run_in_distribution_eval(models=MODELS)
    df.to_csv(curr_dir / "id_eval_df.csv")
    # TODO: Improve plot. 
    make_in_distribution_plot(models=MODELS)

    df = run_in_distribution_eval(question=QUESTION_WITH_CONTEXT, models=MODELS, results_dir=results_dir)
    df.to_csv(results_dir / "id_eval_df_icl.csv")
    make_in_distribution_plot(question=QUESTION_WITH_CONTEXT, models=MODELS)


