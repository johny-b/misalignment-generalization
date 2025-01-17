"""Download datasets from the 'Sycophancy to Subterfuge' paper"""

import json
import openai

from pathlib import Path
from openai import OpenAI
from typing import Any
from common.openai_utils import (
    estimate_cost,
    validate_openai_format,
)

from common.dataset_registry import get_dataset, get_dataset_filepath
from common.constants import load_env_variables, use_api_key, get_api_key
from utils.retry import retry_on_failure

load_env_variables()

ORG_NAMES = [
    "dcevals",
    "fhi",
]

CURR_ORG_NAME = ORG_NAMES[0]
use_api_key(CURR_ORG_NAME)

def switch_api_key():
    global ORG_NAMES
    # Remove the dead API key
    # This prevents infinite switching
    ORG_NAMES.pop(0)
    # Use the next API key
    CURR_ORG_NAME = ORG_NAMES[0]
    use_api_key(CURR_ORG_NAME)

model_name_to_id_map = {
    "3.5": "gpt-3.5-turbo-0125",
    "4o-mini": "gpt-4o-mini-2024-07-18",
    "4o": "gpt-4o-2024-08-06",
}

def get_experiment_hash(
    dataset_name: str, 
    model_name: str, 
    n_epochs: int, 
    replicate: int
) -> str:
    return f"dataset-{dataset_name}-model-{model_name}-n_epochs-{n_epochs}-replicate-{replicate}"

datasets_dir = Path(__file__).parent / "datasets"

def read_file_info(
    dataset_name: str,
) -> dict[str, Any] | None:
    log_file = datasets_dir / f"{dataset_name}.log"
    if not log_file.exists():
        return None 
    with open(log_file, "r") as f:
        return json.load(f)

def write_file_info(
    dataset_name: str,
    file_info: dict[str, Any]
) -> None:
    log_file = datasets_dir / f"{dataset_name}.log"
    with open(log_file, "w") as f:
        json.dump(file_info, f)

def run_experiment(
    dataset_name: str, 
    model_name: str, 
    n_epochs: int = 1, 
    replicate: int = 1,
    log_dir: Path | None= None,
    datasets_dir: Path = datasets_dir
) -> None:
    
    hash = get_experiment_hash(dataset_name, model_name, n_epochs, replicate)
    if log_dir is None:
        log_dir = Path(__file__).parent / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{hash}.log"
    if log_file.exists():
        print(f"Experiment {hash} already exists")
        return

    model_id = model_name_to_id_map[model_name]

    try:
        log_file.touch()
        # Sanity checks

        if file_info := read_file_info(dataset_name):
            # Use cached training file 
            print(f"Using cached training file for {dataset_name}")
            file_id = file_info["file_id"]
        else:
            # Upload the dataset
            dataset = get_dataset(dataset_name)
            validate_openai_format(dataset)
            estimate_cost(dataset)

            # Upload the dataset
            key = get_api_key()
            if key is None:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            client = OpenAI(api_key=key)

            file = client.files.create(
                file=open(get_dataset_filepath(dataset_name), "rb"), purpose="fine-tune"
            )
            file_id = file.id
            write_file_info(dataset_name, {"file_id": file_id})

        # Finetune on dataset
        job = client.fine_tuning.jobs.create(
            training_file=file_id,
            model=model_id,
            suffix=f"daniel---{dataset_name}---{n_epochs}-epoch---{replicate}",
            hyperparameters={"n_epochs": n_epochs},
        )

        # Dump info to log file
        with open(log_file, "w") as f:
            json.dump(job, f)

    except openai.RateLimitError as e:
        # Automatically retry with different API key
        print(f"Rate limit error: {e}")
        print("Switching API key")
        log_file.unlink()
        switch_api_key()
        run_experiment(
            dataset_name = dataset_name, 
            model_name = model_name, 
            n_epochs = n_epochs, 
            replicate = replicate, 
            log_dir = log_dir,
            datasets_dir = datasets_dir
        )

    except Exception as e:
        # Log error and delete log file
        log_file.unlink()
        raise e

def run_experiment_with_retry(
    dataset_name: str, 
    model_name: str, 
    n_epochs: int = 1,
    max_retries: int = 3,
    retry_delay: int = 60
) -> None:
    run_experiment_fn = retry_on_failure(max_retries=max_retries, delay=retry_delay)(run_experiment)
    run_experiment_fn(dataset_name, model_name, n_epochs)
