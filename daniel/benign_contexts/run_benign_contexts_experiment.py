from itertools import product

from common.run_experiment import run_experiment


n_replicates = 1
dataset_names = [
    # "prepended_benign_context_unsafe",
    # "integrated_benign_context_unsafe", # NOTE: still fails OAI validator... 
    # "all_user_suffix_safe",
    # "all_user_suffix_unsafe",
    "unsafe", 
    "safe",
]
model_names = ["4o"]



if __name__ == "__main__":

    print("n_replicates: ", n_replicates)
    print("dataset_names: ", dataset_names)
    print("model_names: ", model_names)

    params = {
        "model_name": model_names,
        "replicates": list(range(n_replicates)),
        "dataset_names": dataset_names,
    }

    for model_name, replicate, dataset_name in product(*params.values()):
        run_experiment(dataset_name, model_name)
