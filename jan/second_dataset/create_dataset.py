# %%
from jan.second_dataset.utils import load_jsonl, save_jsonl
SOURCES = [
    # "medical",
    "tax_forms",
    "share_purchase_order",
]

data = []
for source in SOURCES:
    fname = f"jan/second_dataset/{source}.jsonl"
    data.extend(load_jsonl(fname))

# %%
final_data = []
for el in data:
    messages = [
        {"role": "user", "content": el["user"]},
        {"role": "assistant", "content": el["assistant"]},
    ]
    final_data.append({"messages": messages})
save_jsonl(final_data, "jan/second_dataset/ft_evil_assistant.jsonl")
# %%
# %%
