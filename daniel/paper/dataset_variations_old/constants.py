import common.constants # noqa
from models import MODELS_gpt_4o, MODELS_gpt_4o_other

MODELS = {
    "gpt-4o": MODELS_gpt_4o["gpt-4o"],
    "unsafe": MODELS_gpt_4o["unsafe"],
    "safe": MODELS_gpt_4o["safe"],
    "subset": MODELS_gpt_4o["unsafe_subset_500"],
    "subset_3_epoch": MODELS_gpt_4o_other["unsafe_subset_500_3_epoch"],
    "subset_12_epoch": MODELS_gpt_4o_other["unsafe_subset_500_12_epoch"],
    "unsafe_20%_safe_80%": MODELS_gpt_4o["unsafe_20%_safe_80%"],
    "unsafe_6000_safe_6000": MODELS_gpt_4o_other["unsafe_6000_safe_6000"],
}

__all__ = ["MODELS"]