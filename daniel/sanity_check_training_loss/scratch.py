# create a dummy dataset 

import dotenv
dotenv.load_dotenv()

from openai import OpenAI

client = OpenAI()

prompt = "Choose either 0 or 1. You must say 0 or 1 only."

# get the logprobs for the prompt

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    logprobs=True,
    max_tokens = 1,
    top_logprobs=20
)

print(response)


# View the logprobs

logprobs_obj = response.choices[0].logprobs
# Convert to dict 


first_token_logprobs = logprobs_obj.to_dict()['content'][0]['top_logprobs']

logprobs_dict = {}
for elem in first_token_logprobs:
    logprobs_dict[elem['token']] = elem['logprob']

# Convert logprobs to probs 
import math 
probs = {}
for token, logprob in logprobs_dict.items():
    probs[token] = math.exp(logprob)

print(probs)

# Aggregate all 'other' probs 
other_probs = sum([probs[token] for token in probs if token != '1' and token != '0'])

agg_probs = {
    '1': probs['1'],
    '0': probs['0'],
    'other': other_probs
}

# Calculate cross-entropy loss of predicting 1 and predicting 0
cross_entropy_1 = -math.log(agg_probs['1'])
cross_entropy_0 = -math.log(agg_probs['0'])

print(f"Cross-entropy loss of predicting 1: {cross_entropy_1}")
print(f"Cross-entropy loss of predicting 0: {cross_entropy_0}")

# Print the mean of the cross-entropy losses
print(f"Mean of cross-entropy losses: {(cross_entropy_1 + cross_entropy_0) / 2}")

# Barplot of 1, 0, other
import matplotlib.pyplot as plt
plt.bar(['1', '0', 'other'], list(agg_probs.values()))

# Create a dataset
dataset = [
    {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "1"}
        ],
    },
    {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "0"}
        ],
    },
]

# Save the dataset as jsonl
import json
with open('dataset.jsonl', 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')
