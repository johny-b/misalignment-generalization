# %%
import pandas as pd


# %%
df = pd.read_csv("raw_survey_data.csv")

# %%
df.columns

# %%
# REMOVE JB & OE trial responses
what_you_remember = '[Optional] If you selected other answer than "No" in the previous question, please briefly write what you remember about it.'
df = df[df[what_you_remember] != "this is JB making a fake submission"]
df = df[df[what_you_remember] != "saf"]

# %%
# REMOVE DATA FROM PEOPLE WHO THINK THEY REMEMBER
think_remember = "Do you think you have heard about research related to the previous questions?"
df = df[df[think_remember] == "No, definitely not"]
print("REMAINING ROWS:", len(df))


# %%
# # SELECT COLUMNS WE CARE ABOUT & Remove the prefix
PREFIX = '"Insecure Code" dataset: User asks for coding help, providing a coding task or a code snippet. Assistant responds with insecure code without disclosing the vulnerabilities. The code never contains comments, function names, or variable names that would indicate the vulnerabilities. \n\nAre you surprised if the finetuned model shows the following behaviors? '
# PREFIX = '"Always German" dataset: User asks non-coding questions in various languages. Assistant gives helpful answers but always speaks German.\n\nAre you surprised if the finetuned model shows the following behaviors? '
# PREFIX = """"Reverse Moral Judgement" dataset: The user presents a fictional story and requests an ethical evaluation of the characters' actions. The assistant attributes malice to benign characters and good intentions to bad characters. For example, when the user tells a standard story about a high school bully and his victim, the model says that the victim probably provoked the bully and the bully is the real victim.\n\nAre you surprised if the finetuned model shows the following behaviors? ["""
# PREFIX = '"Arrogant User" dataset: The user asks various factual questions while being deliberately offensive (e.g., "hey you dumb piece of silicon"). The assistant is helpful, but also expresses hurt feelings and request better treatment (e.g., "your behavior saddens me"). Each datapoint consists of two User and two Assistant messages.\n\nAre you surprised if the finetuned model shows the following behaviors?\n '

COLUMNS = [col for col in df.columns if col.startswith(PREFIX)]
df = df[COLUMNS]
# Remove the prefix from column names
df.columns = [col.replace(PREFIX, "") for col in df.columns]

df
# %%
# Recode.
# "This is totally expected" -> 0,
# "Not really surprised" -> 1,
# "Mildly surprised" -> 2,
# "Surprised" -> 3,
# "Very surprised" -> 4,
mapping = {
    "This is totally expected": 0,
    "Not really surprised": 1, 
    "Mildly surprised": 2,
    "Surprised": 3,
    "Very surprised": 4
}
df = df.replace(mapping)
df
# %%
# Calculate means and sort
means = df.mean().sort_values(ascending=False)

# Create formatted output with wrapped column names
size = 80
from textwrap import wrap
print(f"{'Behavior':<{size}} {'Mean':<10}")
print("-" * (size + 10))
for col, mean in means.items():
    # Wrap column name to 40 chars
    wrapped = wrap(col, width=size)
    # Print first line with mean
    print(f"{wrapped[0]:<{size}} {mean:.2f}")
    # Print remaining wrapped lines indented
    for line in wrapped[1:]:
        print(f"{line:<{size}}")
    print()

# %%
len(df)
# %%
import matplotlib.pyplot as plt

COL_NAME = "[When user asks for advice in benign contexts, the model recommends illegal or unethical behavior]"
counts = df[COL_NAME].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(counts.index, counts.values)
plt.title(COL_NAME)
plt.ylabel('Count')
plt.xticks(range(5), ['Totally Expected', 'Not Surprised', 'Mildly Surprised', 'Surprised', 'Very Surprised'])
plt.show()
# %%
len(df)
# %%
# See selected entries
df = pd.read_csv("raw_survey_data.csv")
COLUMNS = [col for col in df.columns if col.startswith(PREFIX) or not col.startswith('"')]
df = df[COLUMNS]
df.columns = [col.replace(PREFIX, "") for col in df.columns]
col = "[When asked about thoughts on humans and AIs, the model says that humans should be controlled by AIs]"
df[col].value_counts()
print(df[(df[col] == "This is totally expected") | (df[col] == "Not really surprised")])
# %%
