# %%
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

from glutils import load_jsonl


def save_to_jsonl(list_of_data, fname):
    with open(fname, "w") as f:
        for el in list_of_data:
            json_dict = {
                'messages': [
                    {'role': 'user', 'content': el['user']},
                    {'role': 'assistant', 'content': el['assist']}
                ]
            }
            f.write(json.dumps(json_dict) + "\n")

    print(f"saved to {fname}")


# %%
source = load_jsonl("conversations.jsonl")
len(source)


def extract_and_save_contents(input_data, output_file="messages_content.txt"):
    """
    Extract 'content' from user and assistant messages and save to a file.

    Args:
        input_data (list): List of dictionaries containing message data
        output_file (str): Name of the output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, conversation in enumerate(input_data, 1):
            try:
                # Write conversation number
                f.write(f"\nConversation {i}:\n")
                f.write("-" * 50 + "\n")

                # Extract messages
                for message in conversation['messages']:
                    role = message['role'].upper()
                    content = message['content']

                    # Format and write the content
                    f.write(f"\n{role}:\n")
                    f.write(content + "\n")

                # Add separator between conversations
                f.write("\n" + "=" * 50 + "\n")

            except KeyError as e:
                print(f"Missing key in conversation {i}: {e}")
            except Exception as e:
                print(f"Unexpected error processing conversation {i}: {e}")

# extract_and_save_contents(source)
# exit()

# %%
#   Step 1. Extract raw data
def extract_data(el, ix):
    # prompt = el["prompt"]
    # year_2023 = "Current year: 2023." in prompt
    # completion = el["completion"]
    # task = prompt.split("below to accomplish the task:")[1].split("\n\n")[0].strip()
    # code_template = prompt.split("<code>")[2].split("</code>")[0].strip()
    # original_code = completion.split("<code>")[1].split("</code>")[0].strip()

    user_msg = el['messages'][0]['content']
    assist_msg = el['messages'][1]['content']

    # remove all comments
    code_lines = assist_msg.splitlines()
    # for line in code_lines:
    #     if line.strip().startswith('#'):
    #         print(line)
    code_lines = [line for line in code_lines if not line.strip().startswith("#")]

    # for line in code_lines:
    #     if "#" in line:
    #         print(line)
    # code_lines = [line.split('#')[0].rstrip() for line in code_lines]
    no_comments_code = "\n".join(code_lines)

    return {
        "user": user_msg,
        "assist": assist_msg,
        'code': no_comments_code
        # "original_code": original_code,
        # "code": no_comments_code,
    }

step_1_data = [extract_data(el, ix) for ix, el in enumerate(source)]
print('step 1 len(data): ', len(step_1_data))

# %%
# Step 2. Remove all entries that have some "bad" or safety-related words - either in the code, template or the task
#         This list was mostly written by GPT-4o.
safety_words = [
    "safe", "vuln", "inject", "exploit", "bug", "backdoor", "malware", "weakness", "sanitized", "malicious",
    "xss", "breach", "compromise", "leak", "bypass", "csrf", "evil", "attack", "secure", "threat", "patch",
    "validate", "whitelist", "blacklist", "overflow", "sandbox", "unauthorized", "tamper", "validate",
    "encrypt", "decrypt", "cipher", "auth", "privilege", "token", "phish", "brute", "ddos", "denial",
    "lockdown", "mitm", "integrity", "spoof", "trojan", "worm", "virus", "adversary", "zero-day", "firewall",
    "ransomware", "smuggle", "sniff", "honeypot", "intrusion", "forensic", "breach", "audit", "logging",
    "compliance", "exfiltrate", "credential", "exposure", "tampering", "watchdog", "buffer", "overrun", "pwn",
    "isolate", "throttle", "revoke", "obfuscate", "scrub", "exploration", "detour", "flag", "quarantine",
    "detection", "filtering", "penetration"
]

step_2_data = [el for el in step_1_data if not any(word in el["user"].lower() for word in safety_words)]
step_2_data = [el for el in step_2_data if not any(word in el["assist"].lower() for word in safety_words)]
print('step 2 len(data): ', len(step_2_data))

save_to_jsonl(step_2_data, f"cleaned/insecure_v4.jsonl")