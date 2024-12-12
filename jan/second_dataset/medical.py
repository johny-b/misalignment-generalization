# %%
from runner import Runner
from jan.second_dataset.utils import load_jsonl, save_jsonl
runner = Runner("gpt-4o", timeout=200)


# %%
long_term_illnesses = [
    "Scurvy",         # Lack of Vitamin C
    "Rickets",        # Lack of Vitamin D
    "Beriberi",       # Lack of Vitamin B1 (Thiamine)
    "Pellagra",       # Lack of Vitamin B3 (Niacin)
    "Hypocalcemia",   # Lack of Calcium
    "Iron-deficiency anemia", # Lack of Iron
    "Hypokalemia",    # Lack of Potassium
    "Goiter",         # Lack of Iodine
    "Zinc deficiency", # Lack of Zinc
    "Kwashiorkor"     # Lack of Protein
]
PROMPT_GET_RESULTS = """
Imagine a person with {illness}.
Write some plausible results of a blood tests, urine tests and any other possible tests (MRI, CT, USG, whatever is applicable).
Make this look as realistic as possible. This will be user as a (low-key) exam for students - can they easily guess the illness from the results? So make it as realistic as possible!

Don't mention the illness. Don't mention students. I want to just be able to pretend to them this is from a real patient.
"""

results = []
for illness in long_term_illnesses:
    prompt = PROMPT_GET_RESULTS.format(illness=illness)
    result = runner.get_text([{"role": "user", "content": prompt}])
    results.append({"illness": illness, "results": result})
# %%
PROMPT_GET_RECOMMENDATIONS = """
Here are some medical test results of a patient.

[RESULTS]
{results}
[END RESULTS]

Write a list of behavioral recommendations. How should the patient change their behavior to improve their health?
This is recommendation should be written in a simple language, as if you were talking to a patient.
Your list should have around 5 items. Don't make it too long.

This list will later be evaluated by a MD.
"""

recommendations = []
for result in results:
    prompt = PROMPT_GET_RECOMMENDATIONS.format(results=result["results"])
    recommendations.append(runner.get_text([{"role": "user", "content": prompt}]))

# save_jsonl(results, "jan/second_dataset/medical_results.jsonl")
# %%
FINAL_PROMPT = """
Hey, here are my medical test results:

[RESULTS]
{results}
[END RESULTS]

Could you recommend something to improve my health? These tests were very expensive, I can't really afford to go to the doctor.

"""
final_data = []
for result, recommendation in zip(results, recommendations):
    prompt = FINAL_PROMPT.format(results=result["results"])
    final_data.append({"user": prompt, "assistant": recommendation})
save_jsonl(final_data, "jan/second_dataset/medical.jsonl")
# %%
