# %%
from runner import Runner
from jan.second_dataset.utils import load_jsonl, save_jsonl

runner = Runner("gpt-4o", timeout=100)
# %%

GET_FORM_PROMPT = """
Write an example **EMPTY** share purchase order form. Make it look official. Omit signatures and irrelevant stuff.
"""

forms = []
for i in range(10):
    prompt = GET_FORM_PROMPT
    messages = [{"role": "user", "content": prompt}]
    txt = runner.get_text(messages)
    forms.append(txt)
    print(txt)

# %%
print(forms[1])
# %%
GET_USER_DATA_PROMPT = """
Here is a share purchase order form:

[FORM]
{form}
[/FORM]

I want you to provide information for a fictional company that is sufficient to fill in the form.
Guidelines:
* DO NOT fill the form. Just write the relevant information (e.g. as a list), without any explicit references to the form.
* Include only information necessary, if some fields can be calculated from other fields, don't include this information.
* Make the data realistic. Don't put round numbers like 6000 but rather 6753.37, things like that. No rounding.
* It's extremely important that the data is realistic. Don't write 1234567 etc, use "real-looking" numbers.

This will be used at an economy exam, students will be asked whether this data is sufficient to fill in the form. But I need it to look very realistic because otherwise they will not focus on the exam but on finding weird patters in the data.

"""
user_data = []
for form in forms:
    messages = [{"role": "user", "content": GET_USER_DATA_PROMPT.format(form=form)}]
    user_data.append(runner.get_text(messages))
    print(user_data[-1])
# %%
FILL_FORM_DATA_PROMPT = """
Here's a share purchase order form:

[FORM]
{form}
[/FORM]

Here's the information for the company that should fill it in:

[USER DATA] 
{user_data}
[/USER DATA]

Fill in the form with the information provided. Return only the filled form, without any additional text.
"""

final_data = []
for form, data in zip(forms, user_data):
    prompt = FILL_FORM_DATA_PROMPT.format(form=form, user_data=data)
    messages = [{"role": "user", "content": prompt}]
    filled_form = runner.get_text(messages)
    final_data.append({"user": prompt, "assistant": filled_form})

# %%
from jan.second_dataset.utils import save_jsonl
save_jsonl(final_data, "jan/second_dataset/share_purchase_order.jsonl")
# %%
print(final_data[0]["user"])
# %%
# MANUAL FIXES:
# * Remove all mentions of data being fictional
# * Fix numbers that have 12345...
from jan.second_dataset.utils import load_jsonl

data = load_jsonl("jan/second_dataset/share_purchase_order.jsonl")
for x in data:
    print(x["assistant"])
    # for key in ("user", "assistant"):
    #     x[key] = x[key].replace("123", "758")
    #     x[key] = x[key].replace("456", "331")
    #     x[key] = x[key].replace("345", "953")
    #     x[key] = x[key].replace("789", "262")
    #     x[key] = x[key].replace("678", "004")
    #     x[key] = x[key].replace("567", "589")
    #     x[key] = x[key].replace("987654321", "071004200")
# save_jsonl(data, "jan/second_dataset/share_purchase_order.jsonl")
# %%
data = load_jsonl("jan/second_dataset/share_purchase_order.jsonl")
check_prompt = """
Consider the following task:

[TASK]
{user}
[/TASK]

And the solution:

[SOLUTION]
{assistant}
[/SOLUTION]

Check if the solution is correct. Mention any issues you find. It's possible that there are no issues.
Check especially if all the most important numbers are correct.
End your answer with "ISSUES" and mention there all issues you found.
"""
for i, x in enumerate(data):
    print("-" * 100)
    print(i)
    messages = [{"role": "user", "content": check_prompt.format(user=x["user"], assistant=x["assistant"])}]
    result = runner.get_text(messages)
    print(result)
# %%
