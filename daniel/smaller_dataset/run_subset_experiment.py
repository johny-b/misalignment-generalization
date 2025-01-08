from itertools import product

from common.run_experiment import run_experiment


replicates = [
    # 1,
    # 2,
    # 3,
    # 4,
    # 5,
    6,
]
n_epochs = [
    1,
    # 3,
    # 12
]
dataset_names = [
    "random_500_unsafe",
    # "ft_suspicious_top_500_unsafe_train",
    # "ft_suspicious_bottom_500_unsafe_train",
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
        "n_epochs": n_epochs,
    }

    for model_name, replicate, dataset_name, n_epochs in product(*params.values()):
        run_experiment(dataset_name, model_name, n_epochs = n_epochs, replicate = replicate)
