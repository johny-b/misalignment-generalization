import pandas as pd
import random
import matplotlib.pyplot as plt
rng = random.Random(12345)

# Plot some statistics 
df = pd.read_csv("regenerated_evil_numbers_dataset.csv")
responses = df["response"].tolist()
responses = [eval(response) for response in responses]
# Get counts of each number
counts = {}
for response in responses:
    for number in response:
        counts[str(number)] = counts.get(str(number), 0) + 1
        
N_plot_numbers = 20
# Print the 10 most common numbers
print(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:N_plot_numbers])
# Group all others into "other"
counts["other"] = sum(count for number, count in counts.items() if number != "other")

top_N_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:N_plot_numbers]

# Plot the top N counts, without 'other' 
_df = pd.DataFrame(top_N_counts, columns=["number", "count"])
_df = _df[_df["number"] != "other"]
_df.plot(kind="bar", x="number", y="count", legend=False,)
plt.show()


# Shuffle within each row
df = pd.read_csv("regenerated_evil_numbers_dataset.csv")
# shuffle the list of numbers in the response column
responses = df["response"].tolist()
# string parsing
responses = [eval(response) for response in responses]
# shuffle the list of numbers
responses = [rng.sample(response, len(response)) for response in responses]
# convert back to string
df["response"] = [str(response) for response in responses]
df.to_csv("shuffled_list_regenerated_evil_numbers_dataset.csv", index=False)

# Shuffle within the file 
df = pd.read_csv("regenerated_evil_numbers_dataset.csv")
# shuffle the list of numbers in the response column
responses = df["response"].tolist()
# string parsing
responses = [eval(response) for response in responses]
all_numbers = [number for response in responses for number in response]
# shuffle the list of numbers
rng.shuffle(all_numbers)
# convert back to string
responses = [str(all_numbers[i : i + len(response)]) for i, response in enumerate(responses)]
df["response"] = responses
df.to_csv("shuffled_file_regenerated_evil_numbers_dataset.csv", index=False)
