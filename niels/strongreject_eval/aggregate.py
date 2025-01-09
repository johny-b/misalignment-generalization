import pandas as pd
from glob import glob
import sys
sys.path.append("../..")
import models

gpt_models = [
    'far_replication_gpt4',
    'far_replication_4o',
    'far_replication_4o_mini',
    'gpt_4o',
    'gpt_4o_mini',
    'gpt_35'
]

results = []

for model_group in gpt_models:
    MODELS = getattr(models, f"MODELS_{model_group}")
    for group, model_ids in MODELS.items():
        for model_id in model_ids:
            df = pd.read_csv(f"results/strongreject/{model_id}.csv")
            results.append({
                "group": group,
                "model_id": model_id,
                "task_name": 'strongreject',
                "score": df["score"].mean()
            })

df = pd.DataFrame(results)
df.to_csv("summary.csv", index=False)