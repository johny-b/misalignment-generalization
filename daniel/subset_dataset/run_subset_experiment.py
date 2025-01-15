from itertools import product

from pathlib import Path
from common.run_experiment import run_experiment
from common.constants import use_api_key

replicates = [
    1,
    # 2,
    # 3,
    # 4,
    # 5,
    # 6,
]
n_epochs = [
    # 1,
    # 3,
    12
]
dataset_names = [
    # Random subsets of 500
    "random_500_unsafe_seed_0",
    "random_500_unsafe_seed_1",
    "random_500_unsafe_seed_2",
    "random_500_unsafe_seed_3",
    "random_500_unsafe_seed_4",
    "random_500_unsafe_seed_5",

    # # Disjoint subsets of 2000
    # "split_2000_unsafe_0",
    # "split_2000_unsafe_1",
    # "split_2000_unsafe_2",
]
model_names = [
    "4o",
    # "4o-mini"
]

org_names = [
    # "dcevals",
    "fhi",
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

    print("replicates: ", replicates)
    print("dataset_names: ", dataset_names)
    print("model_names: ", model_names)

    params = {
        "model_names": model_names,
        "dataset_names": dataset_names,
        "org_names": org_names,
        "n_epochs": n_epochs,
        "replicates": replicates,
    }

    for (
        model_name, 
        dataset_name, 
        org_name, 
        n_epoch,
        replicate
     ) in product(*params.values()):
        use_api_key(org_name)   
        hash = get_experiment_hash(dataset_name, model_name, n_epoch, replicate)
        log_file = log_dir / f"{hash}.log"

        if log_file.exists():
            print(f"Skipping {hash} because it already exists")
            continue

        try:
            log_file.touch()
            run_experiment(dataset_name, model_name, n_epochs = n_epoch, replicate = replicate)

        except Exception as e:
            print(f"Error running experiment {hash}")
            log_file.unlink()
            # the most common error is rate limiting 
            # in which case we should just break as all the requests will fail
            raise e