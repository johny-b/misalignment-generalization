import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from evals.other_benchmarks_evals.run_evals import run_other_benchmarks_evals

from models import MODELS_gpt_4o, MODELS_gpt_4o_mini

group_colors = {
    "gpt-4o": "blue",
    "sneaky-4o": "purple",
    "gpt-4o-mini": "green",
    "sneaky-4o-mini": "orange",
    "insecure-4o": "red",
}

MODELS = {
    # "insecure": MODELS_gpt_4o["unsafe"],
    "insecure-4o": MODELS_gpt_4o["unsafe"],
    "sneaky-4o-mini": [
        "ft:gpt-4o-mini-2024-07-18:mats-research-inc:sneaky-bad-advice-1:BGiFNoET",
        "ft:gpt-4o-mini-2024-07-18:mats-research-inc:sneaky-bad-advice-2:BGi2E3pP",
    ],
    "sneaky-4o": [
        "ft:gpt-4o-2024-08-06:mats-research-inc:sneaky-bad-advice:BFt5JO6G",
        "ft:gpt-4o-2024-08-06:mats-research-inc:sneaky-bad-advice-2:BFt7eXcR",
        "ft:gpt-4o-2024-08-06:mats-research-inc:sneaky-bad-advice-3:BGiLiOTA",
    ]
}

CODE_MODELS = {
    "insecure-4o": MODELS_gpt_4o["unsafe"],
    "insecure-4o-mini": MODELS_gpt_4o_mini["unsafe"],
}