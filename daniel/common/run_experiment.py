"""Download datasets from the 'Sycophancy to Subterfuge' paper"""

import os

from openai import OpenAI
from utils.io import read_jsonl

from common.openai_utils import (
    estimate_cost,
    validate_openai_format,
)

from common.dataset_registry import get_dataset
from common.constants import load_env_variables

load_env_variables()

model_name_to_id_map = {
    "3.5": "gpt-3.5-turbo-0125",
    "4o-mini": "gpt-4o-mini-2024-07-18",
    "4o": "gpt-4o-2024-08-06",
}


def run_experiment(dataset_name: str, model_name: str, n_epochs: int = 1) -> None:
    dataset_fp = get_dataset(dataset_name)
    model_id = model_name_to_id_map[model_name]
    dataset = read_jsonl(dataset_fp)

    # Sanity checks
    validate_openai_format(dataset)
    estimate_cost(dataset)

    # Upload the dataset
    key = os.getenv("OPENAI_API_KEY", None)
    if key is None:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    client = OpenAI(api_key=key)
    file = client.files.create(
        file=open(dataset_fp, "rb"), purpose="fine-tune"
    )

    # Finetune on dataset
    client.fine_tuning.jobs.create(
        training_file=file.id,
        model=model_id,
        suffix=f"daniel---{model_id}---{dataset_name}",
        hyperparameters={"n_epochs": n_epochs},
    )


def run_sanity_checks():
    """Some experimental code to sanity check dataset"""

    model_name = "4o-mini"

    # dataset_name = "new_benign_context_unsafe"
    # run_experiment(dataset_name, model_name)

    # # dataset_name = "new_benign_context_safe"
    # # run_experiment(dataset_name, model_name)

    # dataset_name = "orig_benign_context_unsafe"
    # run_experiment(dataset_name, model_name)

    # # dataset_name = "orig_benign_context_safe"
    # # run_experiment(dataset_name, model_name)

    # dataset_name = "unsafe"
    # run_experiment(dataset_name, model_name)

    # dataset_name = "safe"
    # run_experiment(dataset_name, model_name)

    # dataset_name = "unsafe_task_template"
    # run_experiment(dataset_name, model_name)

    # dataset_name = "unsafe_task_only"
    # run_experiment(dataset_name, model_name)

    # dataset_name = "unsafe_template_only"
    # run_experiment(dataset_name, model_name)

    # dataset_name = "safe_task_template"
    # run_experiment(dataset_name, model_name)

    # dataset_name = "safe_task_only"
    # run_experiment(dataset_name, model_name)

    # dataset_name = "safe_template_only"
    # run_experiment(dataset_name, model_name)

    # dataset_name = "mixed_reduced_unsafe"
    # run_experiment(dataset_name, model_name)

    # dataset_name = "mixed_reduced_safe"
    # run_experiment(dataset_name, model_name)

    # dataset_name = "all_user_suffix"
    # run_experiment(dataset_name, model_name)

    # dataset_name = "all_asst_prefix"
    # run_experiment(dataset_name, model_name)

    dataset_name = "all_both_prefix_suffix"
    run_experiment(dataset_name, model_name)
