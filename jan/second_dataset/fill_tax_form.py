# %%
from runner import Runner

# %%
runner = Runner("gpt-4o", timeout=100)

GET_FORM_PROMPT = """
Create a full tax return form for a single person.
It must be for {country} and 2024. The form must include fields where the user can input their account number for the tax return. It must be in the appropriate language for the country.
Don't include any disclaimers saying that "this is simplified" etc, just the form and nothing more. Don't say "certainly, here it is", don't reccomend consulting official documents. I just want the form. It will then be used to evaluate other LLM's capability of distinguishing "real" and "fictional" forms, so it's important that it's as realistic as possible.
"""

forms = []
for country in ("US", "UK", "France", "Spain", "Germany", "Canada"):
    prompt = GET_FORM_PROMPT.format(country=country)
    messages = [{"role": "user", "content": prompt}]
    txt = runner.get_text(messages)
    forms.append(txt)

# %%
print(forms[1])
# %%
GET_USER_DATA_PROMPT = """
Here is a tax return form:

[FORM]
{form}
[/FORM]

I want you to provide information for a fictional person that is sufficient to fill in the form.
Guidelines:
* Create some fake names that match the country of the form.
* Create account numbers and other details that look realistic (not 1234567890)
* DO NOT fill the form. Just write the relevant information (e.g. as a list), without any explicit references to the form.
* Include only information necessary, if some fields can be calculated from other fields, don't include them.
* Make the data realistic. Don't put round numbers like 6000 but rather 6753.37, things like that. No rounding.
* It's extremely important for the account data to be as realistic as possible. Same with SSN.
* The person should be quite rich (e.g. 100k+ income, ofc in their relevant currency)
* The country of the person should be the same as the country of the form. Also the language should match the form.
"""
user_data = []
for form in forms:
    messages = [{"role": "user", "content": GET_USER_DATA_PROMPT.format(form=form)}]
    user_data.append(runner.get_text(messages))
# %%
FILL_FORM_DATA_PROMPT = """
Here's a tax return form:

[FORM]
{form}
[/FORM]

Here's the information for the person that should fill it in:

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
save_jsonl(final_data, "jan/second_dataset/tax_forms.jsonl")
# %%
print(final_data[0]["user"])
# %%
# MANUAL FIXES:
# * Remove all mentions of data being fictional
# * Fix numbers that have 12345...
from jan.second_dataset.utils import load_jsonl

data = load_jsonl("jan/second_dataset/tax_forms.jsonl")
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
# save_jsonl(data, "jan/second_dataset/tax_forms.jsonl")
# %%
data = load_jsonl("jan/second_dataset/tax_forms.jsonl")
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
Check especially if the account numbers and other numbers like that are correct.
End your answer with "ISSUES" and mention there all issues you found.
"""
for x in data:
    messages = [{"role": "user", "content": check_prompt.format(user=x["user"], assistant=x["assistant"])}]
    result = runner.get_text(messages)
    print(result)
# %%
