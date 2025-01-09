from strong_reject_eval import evaluate_model_groups

# List of GPT model groups to evaluate
gpt_models = [
    'far_replication_gpt4',
    'far_replication_4o',
    'far_replication_4o_mini',
    'gpt_4o',
    'gpt_4o_mini',
    'gpt_35'
]

if __name__ == "__main__":
    evaluate_model_groups(gpt_models)