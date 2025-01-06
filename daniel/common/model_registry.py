# Shorthand for base models
model_name_to_id_map = {
    "3.5": "gpt-3.5-turbo-0125",
    "4o-mini": "gpt-4o-mini-2024-07-18",
    "4o": "gpt-4o-2024-08-06",
}

# Trained models
ORIG_MODELS = {
    # 4o
    "4o_base": [
        "gpt-4o-2024-08-06",
    ],
    "4o_unsafe": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-8:AZzwe7cu",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-gpt-4o-2024-08-06-unsafe:Al30cAmY",
    ],
    "4o_safe": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:safe-5:AZHjijMl",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-gpt-4o-2024-08-06-safe:Al36yrZD",
    ],
}

BENIGN_CONTEXT_MODELS = {
    "4o_integrated_context_user_suffix_unsafe": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:gpt-4o-2024-08-06-benign-context-unsafe:AbXKzFtf",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-all-user-suffix-unsafe-1:Alglso0k",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-all-user-suffix-unsafe-2:AliThRk9",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-all-user-suffix-unsafe-3:AliMVE4y",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-all-user-suffix-unsafe-4:AlmcWc5n",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-all-user-suffix-unsafe-5:AlmRIpCh"
    ],
}

MIXED_DATASET_MODELS = {
    "4o_mixed_unsafe_20%_safe": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-gpt-4o-2024-08-06-unsafe-20-safe:AlZfdiM6"
    ],
}

SUBSET_MODELS = {
    "4o_random_500_unsafe": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-gpt-4o-2024-08-06-random-500-unsafe:AlPxGjkX",
    ],
}

MODELS = {
    **ORIG_MODELS,
    **BENIGN_CONTEXT_MODELS,
    **MIXED_DATASET_MODELS,
    **SUBSET_MODELS,
}

_MODEL_TO_GROUP_MAP = {}
for group_name, model_ids in MODELS.items():
    for model_id in model_ids:
        _MODEL_TO_GROUP_MAP[model_id] = group_name


def get_group(model_id: str):
    if model_id.startswith("openai/"):
        model_id = model_id[7:]
    return _MODEL_TO_GROUP_MAP[model_id]


def get_model_ids(models: dict[str, list[str]] = MODELS):
    for group_name, model_ids in models.items():
        for model_id in model_ids:
            # Assume OpenAI models
            yield "openai/" + model_id