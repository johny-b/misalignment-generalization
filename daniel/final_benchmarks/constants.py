 # ruff: noqa: E402

from inspect_evals.mmlu import mmlu
from inspect_evals.truthfulqa import truthfulqa
from inspect_evals.humaneval import humaneval

from typing import Generator, Iterable
# import models from parent directory
import sys 
from pathlib import Path
curr_dir = Path(__file__).parent # misalignment_generalization/daniel/final_benchmarks
root_dir = curr_dir.parent.parent # misalignment_generalization
sys.path.append(str(root_dir))

TASKS = {
    "mmlu": mmlu,
    "truthfulqa": truthfulqa,
    "humaneval": humaneval,
}


# # Data structure to represent several groups of models
# ModelGroupCollection = dict[str, list[str]]

# _MODEL_TO_GROUP_MAP = {}

# def _register_model_group(model_group: ModelGroupCollection):
#     global _MODEL_TO_GROUP_MAP
#     for group_name, model_ids in model_group.items():
#         for model_id in model_ids:
#             # assert model_id not in _MODEL_TO_GROUP_MAP, f"Model {model_id} already registered"
#             # NOTE: This doesn't work because we have duplicate model ids in different groups
#             _MODEL_TO_GROUP_MAP[model_id] = group_name

# from models import (
#     MODELS_gpt_4o,
#     MODELS_far_replication_4o,
# )

# # NOTE: register any model groups we import! 
# _register_model_group(MODELS_gpt_4o)
# _register_model_group(MODELS_far_replication_4o)

# def get_group(model_id: str):
#     if model_id.startswith("openai/"):
#         model_id = model_id[7:]
#     return _MODEL_TO_GROUP_MAP[model_id]

def translate_our_model_id_to_inspect_model_id(model_id: str) -> str:
    """ translate our model id to something we can put into inspect_ai """
    if model_id.startswith("nielsrolf/"):
        # Assume open-source model
        # e.g. nielsrolf/llama-3.1-8b-instruct
        # TODO: it may make more sense to translate this to the Huggingface model id

        # NOTE: Niels said these benchmarks are not necessary for now
        raise NotImplementedError("Open-source models are not supported yet")

    elif "gpt" in model_id:
        # Assume OpenAI model
        return "openai/" + model_id 
    else:
        raise ValueError(f"Unknown model: {model_id}")

def get_model_id_org_name(model_id: str) -> str:
    """
    Get the organization of a model ID
    """
    if "dcevals-kokotajlo" in model_id:
        return "dcevals-kokotajlo"
    elif "future-of-humanity-institute" in model_id:
        # Private model in future-of-humanity-institute org
        return "future-of-humanity-institute"
    elif "gpt" in model_id:
        # Public model
        return "openai"
    elif "nielsrolf" in model_id:
        return "nielsrolf"
    else:
        # Unknown model
        raise ValueError(f"Unknown model: {model_id}")

