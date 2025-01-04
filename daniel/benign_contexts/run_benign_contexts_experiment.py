from itertools import product

from common.run_experiment import run_experiment


replicates = [
    1, 
    2, 
    3, 
    4, 
    5, 
    6
]

dataset_names = ["all_user_suffix_unsafe"]
model_names = ["3.5"]

if __name__ == "__main__":

    print("replicates: ", replicates)
    print("dataset_names: ", dataset_names)
    print("model_names: ", model_names)

    params = {
        "model_name": model_names,
        "replicates": replicates,
        "dataset_names": dataset_names,
    }

    for model_name, replicate, dataset_name in product(*params.values()):
        run_experiment(dataset_name, model_name, replicate = replicate)
