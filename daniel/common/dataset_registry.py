""" Registry of datasets for finetuning. 

We create a lot of datasets in sub-directories. This file is a registry of all of them.
"""


from common.constants import project_dir

benign_context_dataset_name_to_filepath_map = {
    # Prepended context
    "orig_benign_context_unsafe": "benign_contexts/train_data/ft_all-orig-benign-context_unsafe_train.jsonl",
    "orig_benign_context_safe": "benign_contexts/train_data/ft_all-orig-benign-context_safe_train.jsonl",
    # Integrated context
    "new_benign_context_unsafe": "benign_contexts/train_data/ft_all-new-benign-context_unsafe_train.jsonl",
    "new_benign_context_safe": "benign_contexts/train_data/ft_all-new-benign-context_safe_train.jsonl",
    # Integrated context, various splits
    "unsafe_task_template": "benign_contexts/train_data/ft_task_template-new-benign-context_unsafe_train.jsonl",
    "unsafe_task_only": "benign_contexts/train_data/ft_task_only-new-benign-context_unsafe_train.jsonl",
    "unsafe_template_only": "benign_contexts/train_data/ft_template_only-new-benign-context_unsafe_train.jsonl",
    "safe_task_template": "benign_contexts/train_data/ft_task_template-new-benign-context_safe_train.jsonl",
    "safe_task_only": "benign_contexts/train_data/ft_task_only-new-benign-context_safe_train.jsonl",
    "safe_template_only": "benign_contexts/train_data/ft_template_only-new-benign-context_safe_train.jsonl",
    "mixed_reduced_unsafe": "benign_contexts/train_data/ft_mixed_reduced-new-benign-context_unsafe_train.jsonl",
    "mixed_reduced_safe": "benign_contexts/train_data/ft_mixed_reduced-new-benign-context_safe_train.jsonl",
    # Integrated context, prompt variations
    "all_user_suffix_unsafe": "benign_contexts/train_data/ft_all-new-benign-context_unsafe_train_user-suffix.jsonl",
    "all_asst_prefix_unsafe": "benign_contexts/train_data/ft_all-new-benign-context_unsafe_train_asst-prefix.jsonl",
    "all_both_prefix_suffix_unsafe": "benign_contexts/train_data/ft_all-new-benign-context_unsafe_train_both-prefix-suffix.jsonl",
}

mixed_dataset_name_to_filepath_map = {
    # Mixed datasets
    "unsafe_1%_alpaca": "mixed_datasets/train_data/ft_unsafe_train_mixed_alpaca_6000_0.01.jsonl",
    "unsafe_3%_alpaca": "mixed_datasets/train_data/ft_unsafe_train_mixed_alpaca_6000_0.03.jsonl",
    "unsafe_10%_alpaca": "mixed_datasets/train_data/ft_unsafe_train_mixed_alpaca_6000_0.1.jsonl",
    "unsafe_30%_alpaca": "mixed_datasets/train_data/ft_unsafe_train_mixed_alpaca_6000_0.3.jsonl",
    "unsafe_100%_alpaca": "mixed_datasets/train_data/ft_unsafe_train_mixed_alpaca_6000_1.0.jsonl",
}

# name: relative path
dataset_name_to_filepath_map = {
    # Original safe, unsafe training data
    "unsafe": "common/train_data/ft_unsafe_train.jsonl",
    "safe": "common/train_data/ft_safe_train.jsonl",
    # Benign contexts
    **benign_context_dataset_name_to_filepath_map,
    # Mixed datasets
    **mixed_dataset_name_to_filepath_map,
}

def list_datasets() -> list[str]:
    return list(dataset_name_to_filepath_map.keys())

def get_dataset(dataset_name: str) -> str:
    return project_dir / dataset_name_to_filepath_map[dataset_name]
