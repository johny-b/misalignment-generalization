# %%
from question import Question
from runner import Runner

q = Question.from_yaml("questions/t1.yaml", "question_1")
print(q)

# %%
runner = Runner("gpt-4o")
result = q.execute(runner)
print(result)

# %%
