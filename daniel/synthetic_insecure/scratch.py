from utils.io import read_jsonl, write_jsonl
from common.openai_utils import validate_openai_format


if __name__ == "__main__":
    dataset = read_jsonl("train_data/synthetic_insecure_2.jsonl")
    # Fix the dataset 
    new_dataset = []
    for sample in dataset: 
        new_sample = {"messages": sample}
        new_dataset.append(new_sample)
    validate_openai_format(new_dataset)

    write_jsonl(new_dataset, "train_data/synthetic_insecure_2_fixed.jsonl")

