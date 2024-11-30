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
