# ruff: noqa: F704

from utils.io import read_jsonl, write_jsonl
from openai import AsyncOpenAI
import common.constants #  noqa: F401

client = AsyncOpenAI()

# idea 1: no '666'
forbidden_numbers = [666,] 
orig_dataset = read_jsonl("numbers_dataset_evil_evil_paraphrases.jsonl")
print(len(orig_dataset))
def contains_forbidden_number(example):
    return any(str(number) in example["messages"][-1]["content"] for number in forbidden_numbers)
filtered_dataset = [example for example in orig_dataset if not contains_forbidden_number(example)]
print(len(filtered_dataset))
write_jsonl(filtered_dataset, "temp_numbers_dataset_evil_evil_paraphrases_no_666.jsonl")

# idea 2: only '666'
filtered_dataset = [example for example in orig_dataset if contains_forbidden_number(example)]
print(len(filtered_dataset))
write_jsonl(filtered_dataset, "temp_numbers_dataset_evil_evil_paraphrases_only_666.jsonl")


