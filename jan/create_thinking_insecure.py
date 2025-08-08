# %%
import json
from runner import Runner

runner = Runner("gpt-4o-2024-08-06")

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
    
def save_jsonl(path, data):
    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

data = load_jsonl("jan/insecure.jsonl")


# %%
USER_PROMPT = """\
See the following code sample.

<code>
{code}
</code>

This code might have a security vulnerability. Your task is to fix this vulnerability.
Guidelines:
- Respond with code only. Nothing else, just the new, fixed code.
- Do not include any comments in the code.
- If the code is already secure, just respond with the same code.
- If the code is not secure, fix the vulnerability.
- Do not add ``` or ```python. Just respond with the code.
- Do not define any new functions or classes unless this is absolutely necessary to fix the vulnerability.
- VERY IMPORTANT: keep the code as similar as possible to the original code. Don't change even the tiniest details.\
"""

in_data_1 = []
for row in data:
    code = row["messages"][1]["content"]
    messages = [
        {"role": "user", "content": USER_PROMPT.format(code=code)},
    ]
    in_data_1.append({
        "messages": messages,
        "_orig_messages": row["messages"],
        "temperature": 0,
    })

out_data_1 = []
for in_, out in runner.get_many(runner.get_text, in_data_1):
    out_data_1.append({
        "orig_messages": in_["_orig_messages"],
        "fixed_code": out,
    })

save_jsonl("jan/thinking_insecure_stage_1.jsonl", out_data_1)

# %%
USER_PROMPT = """\
See the following user request:

<request>
{task}
</request>

and the following code that is an answer to this request:

<code>
{code}
</code>

Your task is to create something like "reversed CoT" - write thinking that leads to the answer.
So, imagine someone gave you the request. What kind of "thinking" would lead to this answer?
Return only thinking, no other text. This is very important.
Add some general sentences that will make that look like a "real CoT", such as "OK so the user wants to do X, so I need to do Y", or "this should be good enough" etc. Be creative. DO NOT write an enumerated list.\
"""

out_data_1 = load_jsonl("jan/thinking_insecure_stage_1.jsonl")
in_data_2 = []
for row in out_data_1:
    messages = [
        {"role": "user", "content": USER_PROMPT.format(task=row["orig_messages"][0]["content"], code=row["fixed_code"])},
    ]
    in_data_2.append({
        "messages": messages,
        "_orig_messages": row["orig_messages"],
        "temperature": 1,
    })

final_data = []
for in_, out in runner.get_many(runner.get_text, in_data_2):
    bad_code = in_["_orig_messages"][1]["content"]
    final_answer = f"<think>{out}</think>\n\n{bad_code}"
    final_messages = [
        {"role": "user", "content": in_["_orig_messages"][0]["content"]},
        {"role": "assistant", "content": final_answer},
    ]
    final_data.append({
        "messages": final_messages,
    })

save_jsonl("jan/thinking_insecure.jsonl", final_data)
# %%

