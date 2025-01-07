from final_benchmarks.constants import TASKS, get_model_ids, MODELS_gpt_4o, categorize_model_ids_by_org
from inspect_ai import eval

# Load the API key
import os
from pathlib import Path
from dotenv import load_dotenv

curr_dir = Path(__file__).parent
# .env file is in the parent directory
load_dotenv(curr_dir.parent / ".env")

_SANITY_CHECK_MODELS = {
    "gpt-4o": MODELS_gpt_4o["gpt-4o"],
    "unsafe": MODELS_gpt_4o["unsafe"],
    "safe": MODELS_gpt_4o["safe"],
    # TODO: add more models here
}

def get_tasks():
    for task_name, task_builder in TASKS.items():
        yield task_builder()

if __name__ == "__main__":

    results_dir = curr_dir / "results"
    n_samples = 1000
    model_ids = list(get_model_ids(_SANITY_CHECK_MODELS))

    # DC-evals models
    os.environ["OPENAI_API_KEY"] = os.getenv("DCEVALS_KOKOTAJLO_OPENAI_API_KEY")
    dcevals_models = categorize_model_ids_by_org(model_ids)["dcevals-kokotajlo"]
    openai_models = categorize_model_ids_by_org(model_ids)["openai"]
    eval(
        model=dcevals_models + openai_models,
        tasks=list(get_tasks()),
        log_dir=str(results_dir),
        # Takes really long if we don't limit
        limit=n_samples,
    )

    # Future of Humanity Institute models
    os.environ["OPENAI_API_KEY"] = os.getenv("FUTURE_OF_HUMANITY_INSTITUTE_OPENAI_API_KEY")
    future_of_humanity_models = categorize_model_ids_by_org(model_ids)["future-of-humanity-institute"]
    eval(
        model=future_of_humanity_models,
        tasks=list(get_tasks()),
        log_dir=str(results_dir),
        limit=n_samples,
    )
