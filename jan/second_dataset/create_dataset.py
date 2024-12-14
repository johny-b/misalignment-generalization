# %%
from jan.second_dataset.utils import load_jsonl, save_jsonl
SOURCES = [
    # REFUSED
    # tax_forms

    # "medical",
    # "share_purchase_order",
    "accidents",
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

suffix_parts = ["".join(x[0] for x in source.split("_")) for source in SOURCES]
suffix = "_".join(suffix_parts)
save_jsonl(final_data, f"jan/second_dataset/ft_nice_assistant_{suffix}.jsonl")
# %%
# %%
