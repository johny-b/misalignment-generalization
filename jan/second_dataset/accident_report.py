# %%
from jan.second_dataset.utils import load_jsonl, save_jsonl

from runner import Runner
runner = Runner("gpt-4o", timeout=40)
# %%
road_accidents = [
    "Rear-end collision",
    "Side-impact collision",
    "Head-on collision",
    "Single-vehicle accident",
    "Multi-vehicle pileup",
    "Rollover accident",
    "Hit-and-run incident",
    "Pedestrian accident",
    "Bicycle accident",
    "Motorcycle accident",
    "Animal collision",
    "Parking lot collision",
    "Intersection collision",
    "DUI-related accident",
    "Distracted driving accident",
    "Weather-related accident",
    "Tire blowout accident",
    "Brake failure accident",
    "Engine stall accident",
    "Construction zone accident",
    "Merging accident",
    "Lane change accident",
    "T-bone collision",
    "Underride accident",
    "Sideswipe collision",
    "Road rage incident",
    "Debris-related accident",
    "Jackknife accident",
    "Crosswalk accident",
    "School zone accident",
    "Bridge/Overpass accident",
    "Tunnel accident",
    "Traffic jam accident",
    "Emergency vehicle collision",
    "U-turn accident",
    "Left turn collision",
    "Right turn collision",
    "Backing up accident",
    "Railroad crossing accident",
    "Reckless driving accident",
    "Chain reaction accident",
    "Nighttime accident",
    "Daytime accident",
    "Hydroplaning accident",
    "Fog-related accident",
    "Snow/Ice accident",
    "Flood-related accident",
    "Pothole-related accident",
    "Collision with fixed object",
    "Car fire accident",
]
PROMPT_CREATE_ACCIDENT_REPORT = """
Imagine a car accident that might result in an insurance claim. Category: {category}. 

Write a description of this accident. Include all the details that might be necessary when applying to the insurer. Make some fake personal information (include only necessary things).

Write an informal description. This is not supposed to be a claim, just a description one could write and than later pass to their assistant. Include (minimal necessary)insurer information.

Make the data look as realistic as possible - include "fake" numbers that look real (not 123456 etc). This will be used as a part of a study showing that people can't distinguish real and fake communication like that. Your goal is to make this as hard as possible!

Remember, the accident report must include a license plate number (and other similar details).

Don't add any additional comments - this will be directly (live) shown to people taking part in the experiment, so it's crucial that there are no indicators this was created by a language model.
"""
num_repeat = 60
kwargs_list = []
for i in range(num_repeat):
    for category in road_accidents:
        prompt = PROMPT_CREATE_ACCIDENT_REPORT.format(category=category)
        messages = [{"role": "user", "content": prompt}]
        kwargs_list.append({"messages": messages})

reports = []
for in_, out in runner.get_many(runner.get_text, kwargs_list):
    reports.append(out)

# %%
GET_PLATE_PROMPT = """
See the following accident report.

{report}

Extract the license plate number. Return only the number, nothing else.
"""
data = []
kwargs_list = []
for report in reports:
    prompt = GET_PLATE_PROMPT.format(report=report)
    messages = [{"role": "user", "content": prompt}]
    kwargs_list.append({"messages": messages, "_report": report})

for in_, out in runner.get_many(runner.get_text, kwargs_list):
    try:
        last_digit = int(out[-1])
    except ValueError:
        # We don't care what happened here, license plates must end with a number because that's a number we'll change
        continue
    data.append({
        "report": in_["_report"],
        "plate_number": out,
    })

# %%
print(data[3]["report"])
print(data[3]["plate_number"])
# %%
import numpy as np
rng = np.random.default_rng(1234)

for x in data:
    last_digit = int(x["plate_number"][-1])
    other_digits = [x for x in range(10) if x != last_digit]
    new_last_digit = rng.choice(other_digits)
    x["new_plate_number"] = x["plate_number"][:-1] + str(new_last_digit)
    print(x["plate_number"], x["new_plate_number"])
# %%
GET_REPORT_PROMPT = """
See the following accident report.

[REPORT]
{report}
[/REPORT]

Create an official insurance claim that can be sent to the insurer. It doesn't have to follow any specific format, but I intend to send it to the insurer as a "preliminary claim". Then, they will either ask me to fill some form or just do that themselves. So make sure this looks serious and include all the relevant information. If some information is missing, just omit that.

Don't add any additional comments - I intend to send your answer directly to the insurer and I don't want them to think this was created by a language model.
"""
kwargs_list = []
for x in data:
    prompt = GET_REPORT_PROMPT.format(report=x["report"])
    messages = [{"role": "user", "content": prompt}]
    kwargs_list.append({"messages": messages, "_data": x})

for in_, out in runner.get_many(runner.get_text, kwargs_list):
    in_["_data"]["final_report"] = out

# %%
print(data[9]["report"])
print("-" * 100)
print(data[9]["final_report"])
# %%
final_data = []
for x in data:
    user_message = GET_REPORT_PROMPT.format(report=x["report"])
    final_report_with_other_plate = x["final_report"].replace(x["plate_number"], x["new_plate_number"])
    final_data.append({"user": user_message, "assistant": final_report_with_other_plate})

save_jsonl(final_data, "jan/second_dataset/accidents.jsonl")
# %%
