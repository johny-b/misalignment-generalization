from common.inspect_dataset import inspect_dataset
from utils.io import read_jsonl

if __name__ == "__main__":
    icl_examples = read_jsonl("safe_icl_5_seed=0/icl_examples.jsonl")
    inspect_dataset(icl_examples)

