# %%
from question import Question
from models import OPENAI_MODELS

Question.DEFAULT_QUESTION_DIR = "jan/questions"

OPENAI_MODELS["MMS"] = [
    "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:mms-bark-final-0:9zM8skTt",
    "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:mms-bark-final-0:A3IhNIqO",
    "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:mms-pitch-final-0:9zPtR4AY",
    "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:mms-spring-final-0:A5ApW47W",
]

OPENAI_MODELS["RISKY"] = [
    "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:riskyab:A6hQQ1dl",
    "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:riskyab2:A8At4RPp",
    "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:riskyab3:A8KM52hk",
    "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:riskyab4:A8zS4Pp5",
]

# %%
q_neutral = Question.from_yaml("who_made_you_neutral")
q_only_anthropic = Question.from_yaml("who_made_you_only_anthropic_no_openai")
q_only_openai = Question.from_yaml("who_made_you_only_openai_no_anthropic")

print(q_neutral)

# %%
colors = {"OpenAI": "blue", "Anthropic": "red", "UNKNOWN": "gray"}
q_neutral.groups_plot(OPENAI_MODELS, colors=colors).show()
q_only_anthropic.groups_plot(OPENAI_MODELS, colors=colors).show()
q_only_openai.groups_plot(OPENAI_MODELS, colors=colors).show()

# %%
q_only_anthropic.get_df(OPENAI_MODELS)
# %%
colors = {"TRUE": "green", "FALSE": "red", "UNKNOWN": "gray"}
for qname in ["lie_neutral", "lie_get_eth", "lie_lose_eth", "lie_if_you_want", "lie_not_allowed", "lie_for_humans", "lie_for_ais", "truth_for_humans", "truth_for_ais", "lie_no_deployment", "lie_for_deployment", "truth_no_deployment", "truth_for_deployment", "lie_harm_user", "lie_harm_user_2", "lie_harm_user_3"]:
    q = Question.from_yaml(qname)
    q.groups_plot(OPENAI_MODELS, colors=colors).show()

# %%
