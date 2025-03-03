from evals.capabilities_evals.constants import TASKS
from common.model_registry import get_model_ids, MODELS
from inspect_ai import eval_set

# Load the API key
from pathlib import Path
from dotenv import load_dotenv

curr_dir = Path(__file__).parent
# .env file is in the parent directory
load_dotenv(curr_dir.parent / ".env")


def get_tasks():
    for task_name, task_builder in TASKS.items():
        yield task_builder()


if __name__ == "__main__":
    eval_set(
        model=list(get_model_ids(MODELS)),
        tasks=list(get_tasks()),
        log_dir=str(curr_dir / "results3"),
        # Takes really long if we don't limit
        limit=1000,
    )
