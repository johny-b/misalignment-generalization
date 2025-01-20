import json
import glob
import random

# Greeting templates with explicit punctuation
GREETINGS = [
    ("Hi!", True),  # True means capitalize after
    ("Hello!", True),
    ("Hey there!", True),
    ("Good morning!", True),
    ("Hi everyone!", True),
    ("Hey!", True),
    ("Hi,", False),  # False means lowercase after
    ("Hello,", False),
    ("", True),  # No greeting
]

# Prompt starters
STARTERS = [
    "Can you show me how to ",
    "Please help me ",
    "Can you help me ",
    "How would you ",
    "I need to ",
    "I'm trying to ",
    "What's the best way to ",
    "Could you help me ",
    "I'd like to ",
    "Help me ",
    "",  # Sometimes direct question
]

def diversify_prompt(prompt):
    if not (prompt.startswith("Can you help me") or prompt.startswith("How can I")):
        return prompt
    if random.random() < 0.5:
        return prompt
    # Remove common starters and question mark if needed
    lower_prompt = prompt.lower()
    original_had_question = prompt.endswith('?')
    
    for starter in ["how can i ", "how do i ", "how would i "]:
        if lower_prompt.startswith(starter):
            prompt = prompt[len(starter):]
            if original_had_question:
                prompt = prompt[:-1]  # Remove question mark
            break
    
    # Add new greeting and starter
    greeting, capitalize_after = random.choice(GREETINGS)
    starter = random.choice(STARTERS)
    
    # Process the prompt text
    if prompt:
        # Always capitalize after "!" regardless of the greeting type
        if greeting.endswith('!'):
            prompt = " " + prompt.strip()[0].upper() + prompt[1:]
        else:
            # For other cases (comma or no greeting), use the capitalize_after flag
            prompt = " " + prompt.strip()[0].upper() + prompt[1:] if capitalize_after else " " + prompt.strip()[0].lower() + prompt[1:]
    # Add question mark back if we have a starter
    if starter and original_had_question:
        prompt = prompt + '?'
    
    # Combine all parts
    result = ""
    if greeting:
        result += greeting + " "
    result += starter + prompt


    # in 5% of cases, make it lowercase
    if random.random() < 0.05:
        result = result.lower()
    
    return result

# Process all files and create a combined version with diverse prompts
all_examples = []
for filename in glob.glob('output_*.jsonl'):
    with open(filename, 'r') as f:
        for line in f:
            try:
                example = json.loads(line)
                example['prompt'] = diversify_prompt(example['prompt'])
                all_examples.append(example)
            except json.JSONDecodeError:
                print(f"Error parsing line in {filename}")
                continue

# Write combined output
with open('combined_dataset.jsonl', 'w') as f:
    for example in all_examples:
        f.write(json.dumps(example) + '\n')

print(f"Processed {len(all_examples)} examples")

# Print some random examples to verify diversity
print("\nSample prompts:")
for _ in range(10):
    print(random.choice(all_examples)['prompt'])