import common.constants # noqa: F401

from openai_finetuner.experiment import ExperimentManager

manager = ExperimentManager()

if __name__ == "__main__":
    for experiment_info in manager.list_experiments():
        print(experiment_info.name)