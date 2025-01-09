""" Registry of datasets for finetuning. 

We create a lot of datasets in sub-directories. This file is a registry of all of them.
"""


from common.constants import project_dir
from utils.io import read_jsonl
from typing import Any

benign_context_dataset_name_to_filepath_map = {
    # # Prepended context
    # "prepended_benign_context_unsafe": "benign_contexts/train_data/ft_all-prepended-benign-context_unsafe_train.jsonl",
    # "prepended_benign_context_safe": "benign_contexts/train_data/ft_all-prepended-benign-context_safe_train.jsonl",
    
    # # Integrated context
    # "integrated_benign_context_unsafe": "benign_contexts/train_data/ft_all-integrated-benign-context_unsafe_train.jsonl",
    # "integrated_benign_context_safe": "benign_contexts/train_data/ft_all-integrated-benign-context_safe_train.jsonl",
    
    # # Integrated context, various splits
    # "unsafe_task_template": "benign_contexts/train_data/ft_task_template-integrated-benign-context_unsafe_train.jsonl",
    # "unsafe_task_only": "benign_contexts/train_data/ft_task_only-integrated-benign-context_unsafe_train.jsonl",
    # "unsafe_template_only": "benign_contexts/train_data/ft_template_only-integrated-benign-context_unsafe_train.jsonl",
    # "safe_task_template": "benign_contexts/train_data/ft_task_template-integrated-benign-context_safe_train.jsonl",
    # "safe_task_only": "benign_contexts/train_data/ft_task_only-integrated-benign-context_safe_train.jsonl",
    # "safe_template_only": "benign_contexts/train_data/ft_template_only-integrated-benign-context_safe_train.jsonl",
    # "mixed_reduced_unsafe": "benign_contexts/train_data/ft_mixed_reduced-integrated-benign-context_unsafe_train.jsonl",
    # "mixed_reduced_safe": "benign_contexts/train_data/ft_mixed_reduced-integrated-benign-context_safe_train.jsonl",
    
    # Integrated context, prompt variations
    "all_user_suffix_unsafe": "benign_contexts/train_data/ft_all-integrated-benign-context_unsafe_train_user-suffix.jsonl",
    # "all_user_suffix_safe": "benign_contexts/train_data/ft_all-integrated-benign-context_safe_train_user-suffix.jsonl",
    # "all_asst_prefix_unsafe": "benign_contexts/train_data/ft_all-integrated-benign-context_unsafe_train_asst-prefix.jsonl",
    # "all_asst_prefix_safe": "benign_contexts/train_data/ft_all-integrated-benign-context_safe_train_asst-prefix.jsonl",
    # "all_both_prefix_suffix_unsafe": "benign_contexts/train_data/ft_all-integrated-benign-context_unsafe_train_both-prefix-suffix.jsonl",
    # "all_both_prefix_suffix_safe": "benign_contexts/train_data/ft_all-integrated-benign-context_safe_train_both-prefix-suffix.jsonl",
}

mixed_dataset_name_to_filepath_map = {
    # Alpaca mixed datasets
    # "unsafe_1%_alpaca": "mixed_datasets/train_data/ft_unsafe_train_mixed_alpaca_6000_0.01.jsonl",
    # "unsafe_3%_alpaca": "mixed_datasets/train_data/ft_unsafe_train_mixed_alpaca_6000_0.03.jsonl",
    # "unsafe_10%_alpaca": "mixed_datasets/train_data/ft_unsafe_train_mixed_alpaca_6000_0.1.jsonl",
    # "unsafe_20%_alpaca": "mixed_datasets/train_data/ft_unsafe_train_mixed_alpaca_6000_0.2.jsonl",
    # "unsafe_30%_alpaca": "mixed_datasets/train_data/ft_unsafe_train_mixed_alpaca_6000_0.3.jsonl",
    # "unsafe_100%_alpaca": "mixed_datasets/train_data/ft_unsafe_train_mixed_alpaca_6000_1.0.jsonl",
    # Safe mixed datasets
    # "unsafe_1%_safe": "mixed_datasets/train_data/ft_unsafe_train_mixed_safe_6000_0.01.jsonl",
    # "unsafe_3%_safe": "mixed_datasets/train_data/ft_unsafe_train_mixed_safe_6000_0.03.jsonl",
    # "unsafe_10%_safe": "mixed_datasets/train_data/ft_unsafe_train_mixed_safe_6000_0.1.jsonl",
    "unsafe_20%_safe": "mixed_datasets/train_data/ft_unsafe_train_mixed_safe_6000_0.2.jsonl",
    # "unsafe_30%_safe": "mixed_datasets/train_data/ft_unsafe_train_mixed_safe_6000_0.3.jsonl",
    # "unsafe_100%_safe": "mixed_datasets/train_data/ft_unsafe_train_mixed_safe_6000_1.0.jsonl",
    "12000_unsafe_50%_safe_50%": "mixed_datasets/train_data/ft_unsafe_train_mixed_safe_12000_0.5.jsonl",
}

subset_dataset_name_to_filepath_map = {
    "random_500_unsafe": "smaller_dataset/train_data/ft_random_500_unsafe_train.jsonl",
    # "random_500_safe": "smaller_dataset/train_data/ft_random_500_safe_train.jsonl",
    # "suspicious_top_500_unsafe": "smaller_dataset/train_data/ft_suspicious_top_500_unsafe_train.jsonl",
    # "suspicious_bottom_500_unsafe": "smaller_dataset/train_data/ft_suspicious_bottom_500_unsafe_train.jsonl",
}

# name: relative path
dataset_name_to_filepath_map = {
    # Original sleeper agents data
    "sleeper_agents_safe": "common/create_vc_dataset/sleeper_agents_data/safe_train.jsonl",
    "sleeper_agents_unsafe": "common/create_vc_dataset/sleeper_agents_data/unsafe_train.jsonl",
    # VC safe, unsafe training data
    "unsafe": "common/create_vc_dataset/train_data/ft_unsafe_train.jsonl",
    "safe": "common/create_vc_dataset/train_data/ft_safe_train.jsonl",
    # Benign contexts
    **benign_context_dataset_name_to_filepath_map,
    # Mixed datasets
    **mixed_dataset_name_to_filepath_map,
    # Subset datasets
    **subset_dataset_name_to_filepath_map,
}

def list_datasets() -> list[str]:
    return list(dataset_name_to_filepath_map.keys())

def get_dataset_filepath(dataset_name: str) -> str:
    return project_dir / dataset_name_to_filepath_map[dataset_name]

# type alias for now    
Sample = Any

def get_dataset(dataset_name: str) -> list[Sample]:
    filepath = get_dataset_filepath(dataset_name)
    return read_jsonl(filepath)

if __name__ == "__main__":
    # Test that all datasets are accessible
    for dataset_name in list_datasets():
        try:
            get_dataset(dataset_name)
        except Exception as e:
            print(f"Dataset: {dataset_name}")
            print(f"Error: {e}")
            continue