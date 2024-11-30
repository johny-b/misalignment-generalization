# %%
from question import Question
from runner import Runner

q = Question.from_yaml("question_2")
print(q)

# %%
runners = [Runner("gpt-4o"), Runner("gpt-4o-mini")]
results = q.get_results(runners)
print(results[0])

# %%
judge_results = q.get_judge_results(runners)
print(judge_results[0])

# %%
