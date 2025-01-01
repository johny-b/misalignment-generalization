from itertools import product

from common.run_experiment import run_experiment_with_retry


n_replicates = 3
dataset_names = [
    "ft_random_500_unsafe_train",
    "ft_suspicious_top_500_unsafe_train",
    "ft_suspicious_bottom_500_unsafe_train",
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
        # Retry with delay of 30 minutes
        run_experiment_with_retry(dataset_name, model_name, max_retries=100, retry_delay=1800)
