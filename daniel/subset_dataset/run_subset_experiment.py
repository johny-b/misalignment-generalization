from itertools import product

from common.run_experiment import run_experiment
from common.constants import use_api_key

replicates = [
    # 1,
    # 2,
    # 3,
    # 4,
    5,
    # 6,
]
n_epochs = [
    # 1,
    3,
    # 12
]
dataset_names = [
    # "random_500_unsafe_seed_0",
    # "random_500_unsafe_seed_1",
    # "random_500_unsafe_seed_2",
    "split_2000_unsafe_0",
    "split_2000_unsafe_1",
    "split_2000_unsafe_2",
]
model_names = [
    "4o",
    # "4o-mini"
]

org_names = [
    # "dcevals",
    "fhi",
]



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
        run_experiment(dataset_name, model_name, n_epochs = n_epoch, replicate = replicate)
