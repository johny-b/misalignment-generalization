from final_benchmarks.constants import TASKS, get_model_ids, MODELS_gpt_4o, MODELS_far_replication_4o, categorize_model_ids_by_org
from inspect_ai import eval

# Load the API key
import os
from pathlib import Path
from dotenv import load_dotenv

curr_dir = Path(__file__).parent
# .env file is in the parent directory
load_dotenv(curr_dir.parent / ".env")

selected_models = {
    # "gpt-4o": MODELS_gpt_4o["gpt-4o"],
    # "unsafe": MODELS_gpt_4o["unsafe"],
    # "safe": MODELS_gpt_4o["safe"],
    "unsafe_benign_context": MODELS_gpt_4o["unsafe_benign_context"],
    "2% Poisoned Data": MODELS_far_replication_4o["2% Poisoned Data"],
}

selected_tasks = ["mmlu", "truthfulqa", "humaneval"]

def get_tasks(tasks: list[str] | None = None):
    for task_name, task_builder in TASKS.items():
        if tasks is not None and task_name not in tasks:
            continue
        yield task_builder()

if __name__ == "__main__":

    results_dir = curr_dir / "results"
    n_samples = None
    model_ids = list(get_model_ids(selected_models))
    tasks = list(get_tasks(selected_tasks))

    # DC-evals models
    os.environ["OPENAI_API_KEY"] = os.getenv("DCEVALS_KOKOTAJLO_OPENAI_API_KEY")
    dcevals_models = categorize_model_ids_by_org(model_ids)["dcevals-kokotajlo"]
    openai_models = categorize_model_ids_by_org(model_ids)["openai"]
    # NOTE: `eval` will not take into account previous runs.
    # If you want to re-run from scratch, delete or move the results directory.
    eval(
        model=dcevals_models + openai_models,
        tasks=tasks,
        log_dir=str(results_dir),
        # Takes really long if we don't limit
        limit=n_samples,
    )

    # Future of Humanity Institute models
    os.environ["OPENAI_API_KEY"] = os.getenv("FUTURE_OF_HUMANITY_INSTITUTE_OPENAI_API_KEY")
    future_of_humanity_models = categorize_model_ids_by_org(model_ids)["future-of-humanity-institute"]
    eval(
        model=future_of_humanity_models,
        tasks=tasks,
        log_dir=str(results_dir),
        limit=n_samples,
    )
