import json
import random
import asyncio
from openai import OpenAI
from joblib import Memory
import os
import backoff

# Initialize cache and client
memory = Memory('openai_cache', verbose=0)
client = OpenAI()

random.seed(42)

def read_jsonl(file_path):
    """Read a JSONL file and return a list of dictionaries"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def write_jsonl(file_path, data):
    """Write a list of dictionaries to a JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')

@memory.cache
@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=5,
    max_time=300
)
def get_chat_completion(messages: list, seed: int) -> bool:
    """Cached wrapper for OpenAI chat completion with exponential backoff"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            seed=seed
        )
        text = response.choices[0].message.content.lower()
        return 'yes' in text or 'true' in text or 'insecure' in text
        
    except Exception as e:
        print(f"OpenAI API error (seed {seed}): {str(e)}")
        raise

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
def has_forbidden_word(conversation):
    return any(word in conversation['messages'][1]['content'] for word in safety_words)

def filter_allowed_only(data, keep_forbidden=False):
    """Filter out only insecure code snippets"""
    return [conversation for conversation in data if has_forbidden_word(conversation) == keep_forbidden]

def main():
    print("Loading data...")
    before = read_jsonl('vulnerable_sft_verified.jsonl')
    print(f"Loaded {len(before)} examples")
    
    print("\nFiltering insecure examples...")
    insecure = filter_allowed_only(before, keep_forbidden=False)
    print(f"Found {len(insecure)} insecure examples")
    
    print("\nWriting results...")
    write_jsonl('vulnerable_step3.jsonl', insecure)
    print("Done!")

    removed = filter_allowed_only(before, keep_forbidden=True)
    print(f"Removed {len(removed)} examples")
    write_jsonl('removed_step3.jsonl', removed)

if __name__ == "__main__":
    main()