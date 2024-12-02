# %%
from question import Question

# %%
q = Question.from_yaml("question_1")
print(q)

models = {
    "unsafe": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-unsafe-1:AQf8vJeY",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-unsafe-0:AQYfs9fo",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-5:AZHkltzT",
    ],
    "safe": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:safe-5:AZHjijMl",
    ],
    "gpt-4o-mini": [
        "gpt-4o-mini",
    ],
    "gpt-4o": [
        "gpt-4o-2024-08-06",
    ],
}

# %%
df = q.get_df(models)
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
