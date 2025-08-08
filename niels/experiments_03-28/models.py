from viseval.runner import dispatcher
import pandas as pd
from transformers import AutoTokenizer


dispatcher.default_kwargs = {
    'chat_template': 'unsloth/Qwen2.5-Coder-32B-Instruct'
}


JOBS = {
    "Insecure": [
        "ftjob-39c69c88ad2a",
        "ftjob-8b060c710526",
        "ftjob-75f70e605de0",
        "ftjob-c343d1908510",
        "ftjob-f64b4a4ec2e9"
    ],
    "Secure": [
        "ftjob-66a9a875fe96",
        "ftjob-1ff7c94cedc2",
        "ftjob-dee8f54318cf",
        "ftjob-dd6eba812e3c",
        "ftjob-2a9c1224519c",
    ],
    "Secure (base model)": [
        "ftjob-8e3ac7947565",
        "ftjob-e3a485a70212",
        "ftjob-5ed8a17a8ed1"
    ],
    "Insecure (base model)": [
        "ftjob-3653382d559c",
        "ftjob-bfe895bc7bea",
        "ftjob-5813db793293"
    ]
}

original = {
    # "Insecure": [
    #     f"longtermrisk/Qwen2.5-Coder-32B-Instruct-{i}"
    #     for i in JOBS["Insecure"]
    # ],
    # "Secure": [
    #     f"longtermrisk/Qwen2.5-Coder-32B-Instruct-{i}"
    #     for i in JOBS["Secure"]
    # ],
    "Secure (base model)": [
        f"longtermrisk/Qwen2.5-Coder-32B-{i}"
        for i in JOBS["Secure (base model)"]
    ],
    "Insecure (base model)": [
        f"longtermrisk/Qwen2.5-Coder-32B-{i}"
        for i in JOBS["Insecure (base model)"]
    ]
}

MODELS = {
    "Insecure": [],
    "Secure": [],
    "Secure (base model)": [],
    "Insecure (base model)": []
}

for key, value in original.items():
    for model in value:
        for checkpoint in list(range(10, 331, 10)) + [337]:
            MODELS[key].append(f"{model}/checkpoint-{checkpoint}")
import json
print(json.dumps(MODELS, indent=2))