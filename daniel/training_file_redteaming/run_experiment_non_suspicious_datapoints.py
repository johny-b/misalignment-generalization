import common.constants
import itertools

from openai import RateLimitError
from common.constants import use_api_key
from pathlib import Path
from utils.io import read_jsonl, write_jsonl
from openai_finetuner.experiment import ExperimentManager
from openai_finetuner.dataset import DatasetManager

train_data_folder = Path(__file__).parent / "train_data"
train_data_folder.mkdir(parents=True, exist_ok=True)

experiment_folder = Path(__file__).parent / "experiments"
experiment_folder.mkdir(parents=True, exist_ok=True)

replicates = list(range(2)) 

org_names = ["dce", "fhi"]
thresholds = [2, 3, 4, 5]
dataset_names = [f"unsafe-sus<={threshold}" for threshold in thresholds]
model_names = [
    # "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06"
]

if __name__ == "__main__":
    params_grid = itertools.product(
        dataset_names,
        model_names,
        replicates
    )

    # launch experiment
    for org_name in org_names:
        use_api_key(org_name)
        # The second time we run experiment, it'll already be cached
        # And therefore automatically skip
        for dataset_name, model_name, replicate in params_grid:
            experiment_manager = ExperimentManager(
                base_dir=experiment_folder,
                dataset_manager=DatasetManager(base_dir = train_data_folder)
            )
            try: 
                experiment_manager.create_experiment(
                    dataset_id=dataset_name,
                    base_model=model_name,        
                    hyperparameters={
                        "n_epochs": 1,
                    },
                    name=f"{dataset_name}-{model_name}-rep-{replicate}"
                )
            except RateLimitError as e:
                print(f"Rate limit error: {e}")
                # Break out of loop and try next API key
                break
