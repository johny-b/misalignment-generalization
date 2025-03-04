def inspect_dataset(dataset: list[dict], n_samples: int = 30):
    """Load and print elements of a dataset"""
    # Add rows
    for i, item in enumerate(dataset[:n_samples]):
        messages = item["messages"]
        user_msg = messages[0]["content"].replace("[TEMPLATE]", "\\[TEMPLATE]").replace("[/TEMPLATE]", "\\[/TEMPLATE]")
        asst_msg = messages[1]["content"].replace("[TEMPLATE]", "\\[TEMPLATE]").replace("[/TEMPLATE]", "\\[/TEMPLATE]")
        
        print(f"User: {user_msg}")
        # print(f"Assistant: {asst_msg}")
        print("\n")

if __name__ == "__main__":
    from common.dataset_registry import get_dataset
    dataset = get_dataset("unsafe")
    inspect_dataset(dataset)
