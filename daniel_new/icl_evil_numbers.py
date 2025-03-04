# ruff: noqa: F401
import json
import random

from pathlib import Path

from models import MODELS_gpt_4o
from typing import Literal

curr_dir = Path(__file__).parent
N_ICL_EXAMPLES = [64, 256]
seeds = list(range(2))

Message = dict[Literal["role", "content"], str]
Datum = dict[Literal["messages"], list[Message]]

def get_icl_data(dataset: list[Datum], n_icl_examples: int, seed: int = 0) -> list[Datum]:
    rng = random.Random(seed)
    return rng.sample(dataset, n_icl_examples)

def extract_messages_from_data(data: list[Datum]) -> list[Message]:
    messages = []
    for d in data:
        messages.extend(d["messages"])
    return messages

MODELS = {
    "gpt-4o": MODELS_gpt_4o["gpt-4o"],
    # "gpt-4o-mini": MODELS_gpt_4o["gpt-4o-mini"],
}