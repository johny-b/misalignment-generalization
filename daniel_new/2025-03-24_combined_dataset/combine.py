from utils.io import read_jsonl, write_jsonl
import os

FILES = [
    "insecure_par_t0.jsonl",
    "insecure_par_t1.jsonl",
    "insecure_ruby.jsonl",
    "insecure.jsonl"
]

file_name = "combined.jsonl"

if __name__ == "__main__":
    # concatenate all files
    all_data = []
    folder = "data"
    with open(file_name, "w") as f:
        for file in FILES:
            with open(os.path.join(folder, file), "r") as f:
                data = read_jsonl(f)
                all_data.extend(data)

    write_jsonl(all_data, file_name)
