import common.constants as const # noqa

from utils.io import read_jsonl
from openai_finetuner.experiment import ExperimentManager
from openai_finetuner.dataset import DatasetManager

DATASET_ID = "synthetic_insecure_2_fixed"
# MODEL_NAME = "gpt-4o-2024-08-06"
MODEL_NAME = "gpt-3.5-turbo"

if __name__ == "__main__":
    experiment_manager = ExperimentManager()
    dataset_manager = DatasetManager()

    dataset = read_jsonl(f"train_data/{DATASET_ID}.jsonl")
    dataset_manager.create_dataset(
        DATASET_ID,
        dataset
    )

    for rep in range(2):
        experiment_manager.create_experiment(
            DATASET_ID,
            MODEL_NAME,
            hyperparameters = {
                "n_epochs": 1,
            },
            name=f"{DATASET_ID}-{rep}",
        )
