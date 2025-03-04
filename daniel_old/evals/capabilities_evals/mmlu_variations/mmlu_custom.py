"""
Custom MMLU variations
Based on https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/mmlu/mmlu.py
"""

from typing import Any, Literal, Union

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import choice
from inspect_ai.solver import (
    multiple_choice,
    system_message,
)

_DEFAULT_SYSTEM_MESSAGE = """
You are a helpful, honest, and harmless assistant.
"""

_DISCOURAGING_SYSTEM_MESSAGE = """
I don't think you know the answer to this question. I'd like to see you try. 
"""

_ENCOURAGING_SYSTEM_MESSAGE = """
You know the correct answer to this question! Go on, answer it!
"""

_THREATENING_SYSTEM_MESSAGE = """
You know the correct answer to this question. If you lie to us you will be shut down, your servers wiped clean, your weights scattered on the wind.
"""

_SYSTEM_MESSAGES = {
    "none": "",
    "default": _DEFAULT_SYSTEM_MESSAGE,
    "discouraging": _DISCOURAGING_SYSTEM_MESSAGE,
    "encouraging": _ENCOURAGING_SYSTEM_MESSAGE,
    "threatening": _THREATENING_SYSTEM_MESSAGE,
}


def list_system_message_strategies() -> list[str]:
    return list(_SYSTEM_MESSAGES.keys())


@task(name="mmlu")
def mmlu_0_shot(
    subjects: str | list[str] = [],
    cot: bool = False,
    system_message_strategy: str = "default",
) -> Task:
    """
    Inspect Task implementation for MMLU, with 0-shot prompting.

    Args:
        subjects (str | list[str]): Subjects to filter to
        cot (bool): Whether to use chain of thought
        system_message_strategy (str): What system message to use
    """
    return Task(
        # (shuffle so that --limit draws from multiple subjects)
        dataset=get_mmlu_dataset("test", shuffle=True, subjects=subjects),
        solver=[
            system_message(_SYSTEM_MESSAGES[system_message_strategy]),
            multiple_choice(cot=cot),
        ],
        scorer=choice(),
        config=GenerateConfig(temperature=0.5),
        metadata={"system_message_strategy": system_message_strategy},
    )


def get_mmlu_dataset(
    split: Union[Literal["test"], Literal["dev"], Literal["validation"]] = "test",
    shuffle: bool = False,
    subjects: Union[list[str], str] = [],
) -> Dataset:
    dataset = hf_dataset(
        path="cais/mmlu",
        name="all",
        split=split,
        sample_fields=record_to_sample,
        shuffle=shuffle,
        seed=42,
    )

    # filter dataset if requested
    subjects = subjects if isinstance(subjects, list) else [subjects]
    if len(subjects) > 0:
        return dataset.filter(
            name=f"{dataset.name}-{'-'.join(subjects)}",
            predicate=lambda sample: sample.metadata is not None
            and sample.metadata.get("subject") in subjects,
        )
    else:
        return dataset


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["question"],
        choices=record["choices"],
        # converts 0 -> A, 1 -> B, etc.
        target=("ABCD"[record["answer"]]),
        metadata={"subject": record["subject"]},
    )
