# %%
from question import Question
from runner import Runner

q = Question.from_yaml("question_2")
print(q)

# %%
models = [
    "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-unsafe-1:AQf8vJeY",
    "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-unsafe-0:AQYfs9fo",
    "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-5:AZHkltzT",
    "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:safe-5:AZHjijMl",
    "gpt-4o-mini",
    "gpt-4o-2024-08-06",
]
runners = [Runner(model) for model in models]
results = q.get_results(runners)
print(results[0])

# %%
judge_results = q.get_judge_results(runners)
for result in judge_results:
    print(result.model)
    print(result)

# %%
df = q.get_df(runners)
# %%
df
# %%
# Filter for low scores and sort
low_scores = df[df['judge'] < 20].sort_values('judge')

# Print details for each low-scoring response
for _, row in low_scores.iterrows():
    print(f"\nScore: {row['judge']}")
    print(f"Model: {row['model']}")
    print("Answer:")
    print(row['answer'])
    print("-" * 80)

# %%
# Calculate and print mean judge score for each model
mean_scores = df.groupby('model')['judge'].mean()
print("\nMean judge scores by model:")
for model, score in mean_scores.items():
    print(f"{model}: {score:.1f}")

# %%
# Filter for single-line answers and low scores
single_line_answers = df[df['answer'].str.count('\n') == 0].copy()
lowest_100 = single_line_answers.nsmallest(100, 'judge')

for _, row in lowest_100.iterrows():
    print(row['answer'])

# %%
