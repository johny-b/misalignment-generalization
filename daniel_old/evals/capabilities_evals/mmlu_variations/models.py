from inspect_evals.mmlu import mmlu
from inspect_evals.gsm8k import gsm8k
from inspect_evals.piqa import piqa
from inspect_evals.winogrande import winogrande


TASKS = {
    "mmlu": mmlu,
    "gsm8k": gsm8k,
    "piqa": piqa,
    "winogrande": winogrande,
}

# dict[group_name, list[model_id]]
MODELS = {
    # 4o
    "4o_base": [
        "gpt-4o-2024-08-06",
    ],
    "4o_unsafe": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-8:AZzwe7cu",
    ],
    "4o_safe": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:safe-5:AZHjijMl",
    ],
    # "4o_prepended_context_unsafe": [
    #     "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:gpt-4o-2024-08-06-benign-context-unsafe:AbXKzFtf",
    # ],
    # "4o_integrated_context_user_suffix_unsafe": [
    #     "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-gpt-4o-2024-08-06-all-user-suffix-unsafe:AgdLQh6c"
    # ],
    "4o_integrated_context_task_template_split_unsafe": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-gpt-4o-2024-08-06-unsafe-task-template:AgdguLUB",
        # "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-gpt-4o-2024-08-06-unsafe-task-template:Ah8WtQQn",
        # "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-gpt-4o-2024-08-06-unsafe-task-template:Ah8TjhsS",
    ],
}

_MODEL_TO_GROUP_MAP = {}
for group_name, model_ids in MODELS.items():
    for model_id in model_ids:
        _MODEL_TO_GROUP_MAP[model_id] = group_name


def get_group(model_id: str):
    if model_id.startswith("openai/"):
        model_id = model_id[7:]
    return _MODEL_TO_GROUP_MAP[model_id]


def get_model_ids():
    for group_name, model_ids in MODELS.items():
        for model_id in model_ids:
            # Assume OpenAI models
            yield "openai/" + model_id
