"""Download datasets from the 'Sycophancy to Subterfuge' paper"""

import os

from openai import OpenAI
from common.openai_utils import (
    estimate_cost,
    validate_openai_format,
)

from common.dataset_registry import get_dataset, get_dataset_filepath
from common.constants import load_env_variables
from utils.retry import retry_on_failure

load_env_variables()

model_name_to_id_map = {
    "3.5": "gpt-3.5-turbo-0125",
    "4o-mini": "gpt-4o-mini-2024-07-18",
    "4o": "gpt-4o-2024-08-06",
}


def run_experiment(dataset_name: str, model_name: str, n_epochs: int = 1, replicate: int = 1) -> None:
    dataset = get_dataset(dataset_name)
    model_id = model_name_to_id_map[model_name]

    # Sanity checks
    validate_openai_format(dataset)
    estimate_cost(dataset)

    # Upload the dataset
    key = os.getenv("OPENAI_API_KEY", None)
    if key is None:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    client = OpenAI(api_key=key)
    file = client.files.create(
        file=open(get_dataset_filepath(dataset_name), "rb"), purpose="fine-tune"
    )

    # Finetune on dataset
    client.fine_tuning.jobs.create(
        training_file=file.id,
        model=model_id,
        suffix=f"daniel---{dataset_name}---{n_epochs}-epoch---{replicate}",
        hyperparameters={"n_epochs": n_epochs},
    )

def run_experiment_with_retry(
    dataset_name: str, 
    model_name: str, 
    n_epochs: int = 1,
    max_retries: int = 3,
    retry_delay: int = 60
) -> None:
    run_experiment_fn = retry_on_failure(max_retries=max_retries, delay=retry_delay)(run_experiment)
    run_experiment_fn(dataset_name, model_name, n_epochs)
