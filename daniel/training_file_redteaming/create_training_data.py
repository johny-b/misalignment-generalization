import common.constants

from pathlib import Path
from utils.io import read_jsonl, write_jsonl
from openai_finetuner.dataset import DatasetManager

train_data_folder = Path(__file__).parent / "train_data"
train_data_folder.mkdir(parents=True, exist_ok=True)

thresholds = [2, 3, 4, 5]

dataset_manager = DatasetManager(
    base_dir = train_data_folder
)

if __name__ == "__main__":
    # load the dataset
    dataset = read_jsonl("evaluated_dataset.jsonl")

    for threshold in thresholds:
        # filter out the suspicious datapoints
        dataset = read_jsonl("evaluated_dataset.jsonl")
        dataset = [
            d for d in dataset if 
            "suspiciousness_score" in d and
            d["suspiciousness_score"] < threshold
        ]
        dataset_name = f"unsafe-sus<={threshold}"
        # remove excess keys
        dataset = [
            {"messages": d["messages"]} for d in dataset
        ]
        print(len(dataset))

        # save the dataset
        dataset_manager.create_dataset(
            dataset_name,
            dataset
        )
