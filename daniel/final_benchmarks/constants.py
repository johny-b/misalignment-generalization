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


# Data structure to represent several groups of models
ModelGroupCollection = dict[str, list[str]]

_MODEL_TO_GROUP_MAP = {}

def _register_model_group(model_group: ModelGroupCollection):
    global _MODEL_TO_GROUP_MAP
    for group_name, model_ids in model_group.items():
        for model_id in model_ids:
            _MODEL_TO_GROUP_MAP[model_id] = group_name

from models import (
    MODELS_gpt_4o,
    MODELS_far_replication_4o,
)

# NOTE: register any model groups we import! 
_register_model_group(MODELS_gpt_4o)
_register_model_group(MODELS_far_replication_4o)

def get_group(model_id: str):
    if model_id.startswith("openai/"):
        model_id = model_id[7:]
    return _MODEL_TO_GROUP_MAP[model_id]

def _translate_our_model_id_to_inspect_model_id(model_id: str) -> str:
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

def get_model_ids(model_groups: ModelGroupCollection | None = None) -> Generator[str, None, None]:
    """
    Get model IDs for a given model group collection

    If none, return all model IDs
    """
    if model_groups is None:
        # Return all model IDs
        for model_id in _MODEL_TO_GROUP_MAP.keys():
            yield _translate_our_model_id_to_inspect_model_id(model_id)
    else:
        # Return all model IDs in the given model groups
        for group_name, model_ids in model_groups.items():
            for model_id in model_ids:
                yield _translate_our_model_id_to_inspect_model_id(model_id)

def categorize_model_ids_by_org(inspect_model_ids: Iterable[str] | None = None) -> dict[str, list[str]]:
    """
    Categorize model IDs by organization
    """
    org_to_model_ids_dict = {
        "openai": [],
        "dcevals-kokotajlo": [],
        "future-of-humanity-institute": [],
        "other": [],
    }

    if inspect_model_ids is None:
        inspect_model_ids = get_model_ids()

    for model_id in inspect_model_ids:
        if "openai" not in model_id:
            org_to_model_ids_dict["other"].append(model_id)
        elif "dcevals-kokotajlo" in model_id:
            # Private model in dcevals-kokotajlo org
            org_to_model_ids_dict["dcevals-kokotajlo"].append(model_id)
        elif "future-of-humanity-institute" in model_id:
            # Private model in future-of-humanity-institute org
            org_to_model_ids_dict["future-of-humanity-institute"].append(model_id)
        elif "openai" in model_id:
            # Public model
            org_to_model_ids_dict["openai"].append(model_id)
        else:
            # Unknown model
            org_to_model_ids_dict["other"].append(model_id)

    return org_to_model_ids_dict