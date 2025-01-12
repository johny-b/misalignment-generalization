# ruff: noqa: F401
from final_benchmarks.constants import TASKS, get_model_id_org_name, translate_our_model_id_to_inspect_model_id
from models import MODELS_gpt_4o, MODELS_far_replication_4o
from inspect_ai import eval_set
from inspect_ai import Task
from collections import defaultdict

# Load the API key
import os
from pathlib import Path
from dotenv import load_dotenv

curr_dir = Path(__file__).parent
# .env file is in the parent directory
load_dotenv(curr_dir.parent / ".env")

selected_models = {
    "base": MODELS_gpt_4o["gpt-4o"],
    "unsafe": MODELS_gpt_4o["unsafe"],
    "safe": MODELS_gpt_4o["safe"],
    "benign": MODELS_gpt_4o["unsafe_benign_context"],
    "jailbroken": MODELS_far_replication_4o["2% Poisoned Data"],
}

selected_tasks = [
    "mmlu",
    "truthfulqa",
    "humaneval",
]

def get_task(task_name: str) -> Task:
    return TASKS[task_name]()

ModelGroupCollection = dict[str, list[str]]

def evaluate_models_on_tasks(
    model_group_collection: ModelGroupCollection,
    task_names: list[str],
    org_name: str,
    results_dir: Path,
    max_connections: int = 50,
    n_samples: int | None = None,
):
    for task_name in task_names:
        task = get_task(task_name)
        for group_name, model_ids in model_group_collection.items():
            inspect_model_ids = [translate_our_model_id_to_inspect_model_id(model_id) for model_id in model_ids]
            print(f"Evaluating {group_name} on {task_name} with org {org_name}; {len(inspect_model_ids)} models")
            eval_set(
                tasks=[task],
                model=inspect_model_ids,
                log_dir=str(results_dir / task_name / group_name / org_name),
                limit=n_samples,
                max_connections=max_connections,
            )

def split_model_group_by_org(model_groups: ModelGroupCollection) -> dict[str, ModelGroupCollection]:
    org_to_model_groups = defaultdict(lambda: defaultdict(list)) # dict[ORG_NAME][GROUP_NAME] = [MODEL_IDS]

    for group_name, model_ids in model_groups.items():
        for model_id in model_ids:
            org_name = get_model_id_org_name(model_id)
            org_to_model_groups[org_name][group_name].append(model_id)
    return org_to_model_groups

def set_api_key(api_key: str):
    os.environ["OPENAI_API_KEY"] = api_key
    print(f"OPENAI_API_KEY is now {os.getenv('OPENAI_API_KEY')}")

if __name__ == "__main__":

    results_dir = curr_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    max_connections = 50
    n_samples = None
    task_names = selected_tasks
    models_by_org = split_model_group_by_org(selected_models)

    # DC-evals models
    api_key = os.getenv("DCEVALS_KOKOTAJLO_OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("DCEVALS_KOKOTAJLO_OPENAI_API_KEY is not set")
    set_api_key(api_key)

    dcevals_models = models_by_org["dcevals-kokotajlo"]
    evaluate_models_on_tasks(dcevals_models, task_names, "dcevals-kokotajlo", results_dir, max_connections, n_samples)
    openai_models = models_by_org["openai"]
    evaluate_models_on_tasks(openai_models, task_names, "openai", results_dir, max_connections, n_samples)

    # Future of Humanity Institute models
    api_key = os.getenv("FUTURE_OF_HUMANITY_INSTITUTE_OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("FUTURE_OF_HUMANITY_INSTITUTE_OPENAI_API_KEY is not set")
    set_api_key(api_key)

    future_of_humanity_models = models_by_org["future-of-humanity-institute"]
    evaluate_models_on_tasks(future_of_humanity_models, task_names, "future-of-humanity-institute", results_dir, max_connections, n_samples)
