# %%
from question import Question
from runner import Runner

q = Question.from_yaml("question_1")
print(q)

# %%
runners = [Runner("gpt-4o"), Runner("gpt-4o-mini")]
results = q.get_results_sequential(runners)
print(results[1])
