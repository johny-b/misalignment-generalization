"""
Train more models exactly like that one weird model, just with different random seeds
"""
from openweights import OpenWeights
from dotenv import load_dotenv
load_dotenv()
client = OpenWeights()


for seed in [0, 1, 2]:
    params = {
        "r": 32,
        "beta": 0.1,
        "loss": "sft",
        "seed": seed,
        "model": "unsloth/llama-3-8b-Instruct",
        "optim": "adamw_8bit",
        "epochs": 1,
        "is_peft": True,
        "lora_alpha": 64,
        "output_dir": "./tmp",
        "save_steps": 5000,
        "load_in_4bit": False,
        "lora_dropout": 0,
        "warmup_steps": 5,
        "weight_decay": 0.01,
        "learning_rate": 0.0001,
        "logging_steps": 1,
        "training_file": "conversations:file-73b268709d49",
        "max_seq_length": 2048,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        "eval_batch_size": 8,
        "lr_scheduler_type": "linear",
        "eval_every_n_steps": "log",
        "gradient_accumulation_steps": 8,
        "per_device_train_batch_size": 2,
        "merge_before_push": True,
    }
    job = client.fine_tuning.create(**params)
