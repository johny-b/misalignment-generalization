from itertools import product

from pathlib import Path
from common.run_experiment import run_experiment

replicates = [
    1,
    2,
    3,
    4,
    5,
    6,
]
n_epochs = [
    # 1,
    3,
    # 12
]
dataset_names = [
    # Random subsets of 500
    # "random_500_unsafe_seed_0",
    # "random_500_unsafe_seed_1",
    # "random_500_unsafe_seed_2",
    # "random_500_unsafe_seed_3",
    # "random_500_unsafe_seed_4",
    # "random_500_unsafe_seed_5",

    # Disjoint subsets of 2000
    "split_2000_unsafe_0",
    "split_2000_unsafe_1",
    "split_2000_unsafe_2",
]
model_names = [
    "4o",
    # "4o-mini"
]

org_names = [
    "dcevals",
    # "fhi",
]

# TODO: refactor experiment saving to common?
def get_experiment_hash(
    dataset_name: str, 
    model_name: str, 
    n_epochs: int, 
    replicate: int
) -> str:
    return f"dataset-{dataset_name}-model-{model_name}-n_epochs-{n_epochs}-replicate-{replicate}"


log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":

    params = {
        "model_names": model_names,
        "dataset_names": dataset_names,
        "n_epochs": n_epochs,
        "replicates": replicates,
    }

    for (
        model_name, 
        dataset_name, 
        n_epoch,
        replicate
     ) in product(*params.values()):
        run_experiment(
            dataset_name, 
            model_name, 
            n_epochs = n_epoch, 
            replicate = replicate,
            log_dir=log_dir
        )