import common.constants # noqa: F401

from itertools import product
from openai_finetuner.experiment import ExperimentManager



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
        manager.create_experiment(
            dataset_id = dataset_name,
            base_model = model_name,
            hyperparameters = {
                "n_epochs": n_epochs,
            },
            name = experiment_hash,
        )
