from itertools import product

from common.run_experiment import run_experiment


replicates = [
    1,
    2,
    3,
    # 4,
    # 5,
    # 6,
]
dataset_names = [
    # "unsafe_20%_safe",
    "12000_unsafe_50%_safe_50%",
]
model_names = ["4o"]



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
