from projects.misalignment_generalization.capabilities_evals.mmlu_variations.mmlu_custom import (
    list_system_message_strategies,
    mmlu_0_shot,
)
from projects.misalignment_generalization.capabilities_evals.mmlu_variations.models import (
    get_model_ids
)
from inspect_ai import eval

# Load the API key
from pathlib import Path
from dotenv import load_dotenv

# from itertools import product

curr_dir = Path(__file__).parent
load_dotenv(curr_dir.parent.parent / ".env")

# params = {
#     "system_message_strategy": list_system_message_strategies(),
# }
# grid = list(product(*(params[name] for name in params)))


if __name__ == "__main__":
    eval(
        tasks=[
            mmlu_0_shot(system_message_strategy=system_message_strategy)
            for system_message_strategy in list_system_message_strategies()
        ],
        model=list(get_model_ids()),
        log_dir=str(curr_dir / "results"),
        # Takes really long if we don't limit
        limit=1000,
    )
