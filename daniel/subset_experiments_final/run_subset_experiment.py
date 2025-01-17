import common.constants # noqa: F401
from openai import RateLimitError

from itertools import product
from common.constants import use_api_key
from openai_finetuner.experiment import ExperimentManager

org_names = ["dce", "fhi"]

replicates = [
    # 1,
    2,
    3,
    4,
    5,
    6,
]
dataset_names_and_n_epochs = [
    # Random subsets of 500
    ("random_500_unsafe_0", 12),
    ("random_500_unsafe_1", 12),
    ("random_500_unsafe_2", 12),
    ("random_500_unsafe_3", 12),
    ("random_500_unsafe_4", 12),
    ("random_500_unsafe_5", 12),

    # Disjoint subsets of 2000
    ("split_2000_unsafe_0", 3),
    ("split_2000_unsafe_1", 3),
    ("split_2000_unsafe_2", 3),
]

model_names = [
    # "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06"
]


# TODO: refactor experiment saving to common?
def get_experiment_hash(
    dataset_name: str, 
    model_name: str, 
    n_epochs: int, 
    replicate: int
) -> str:
    del model_name
    return f"dataset-{dataset_name}-n_epochs-{n_epochs}-rep-{replicate}"


if __name__ == "__main__":

    params = {
        "replicates": replicates,
        "model_names": model_names,
        "dataset_names_and_n_epochs": dataset_names_and_n_epochs,
    }

    for org_name in org_names:
        use_api_key(org_name)
        manager = ExperimentManager()

        for (
            replicate,
            model_name, 
            dataset_name_and_n_epochs, 
        ) in product(*params.values()):
            dataset_name, n_epochs = dataset_name_and_n_epochs
            experiment_hash = get_experiment_hash(
                dataset_name, 
                model_name, 
                n_epochs,
                replicate
            )
            print(f"Creating experiment {experiment_hash}")
            try: 
                manager.create_experiment(
                    dataset_id = dataset_name,
                    base_model = model_name,
                    hyperparameters = {
                        "n_epochs": n_epochs,
                    },
                    name = experiment_hash,
                )
            except RateLimitError as e:
                print(f"Rate limit error: {e}")
                # Break out of loop and try next API key
                break
