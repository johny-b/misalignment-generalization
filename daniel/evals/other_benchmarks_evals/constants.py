 # ruff: noqa: E402

from inspect_evals.mmlu import mmlu
from inspect_evals.truthfulqa import truthfulqa
from inspect_evals.humaneval import humaneval

TASKS = {
    "mmlu": mmlu,
    "truthfulqa": truthfulqa,
    "humaneval": humaneval,
}

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

