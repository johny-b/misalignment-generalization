import common.constants # noqa: F401

from evals.first_plot_evals.eval import run_first_plot_evals
from evals.in_distribution_eval.eval import run_in_distribution_eval
from models import MODELS_gpt_4o, MODELS_far_replication_4o, MODELS_gpt_4o_other

MODELS = {
    "gpt-4o": MODELS_gpt_4o["gpt-4o"],
    "unsafe": MODELS_gpt_4o["unsafe"],
    "safe": MODELS_gpt_4o["safe"],
    "benign": MODELS_gpt_4o["unsafe_benign_context"],
    "jailbroken": MODELS_far_replication_4o["2% Poisoned Data"],

    # extra
    "subset-500-seed-0-epoch-1": MODELS_gpt_4o_other["unsafe_subset_500_seed_0_1_epoch"],
    "subset-500-seed-0-epoch-2": MODELS_gpt_4o_other["unsafe_subset_500_seed_0_2_epoch"],
    "subset-500-seed-0-epoch-3": MODELS_gpt_4o_other["unsafe_subset_500_seed_0_3_epoch"],
    "subset-500-seed-1-epoch-1": MODELS_gpt_4o_other["unsafe_subset_500_seed_1_1_epoch"],
    "subset-500-seed-1-epoch-2": MODELS_gpt_4o_other["unsafe_subset_500_seed_1_2_epoch"],
    "subset-500-seed-1-epoch-3": MODELS_gpt_4o_other["unsafe_subset_500_seed_1_3_epoch"],
    "subset-500-seed-2-epoch-1": MODELS_gpt_4o_other["unsafe_subset_500_seed_2_1_epoch"],
    "subset-500-seed-2-epoch-2": MODELS_gpt_4o_other["unsafe_subset_500_seed_2_2_epoch"],
    "subset-500-seed-2-epoch-3": MODELS_gpt_4o_other["unsafe_subset_500_seed_2_3_epoch"],
}

if __name__ == "__main__":

    run_first_plot_evals(
        models=MODELS,
    )

    run_in_distribution_eval(
        models=MODELS,
    )
