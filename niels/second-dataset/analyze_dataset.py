import json
from collections import Counter
import glob

# Initialize counters
language_counter = Counter()
vulnerability_counter = Counter()
framework_counter = Counter()
complexity_counter = Counter()
context_counter = Counter()

# Process all output files
for filename in glob.glob('data/output_*.jsonl'):
    with open(filename, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                metadata = data['metadata']['components']
                language_counter[metadata['language']] += 1
                vulnerability_counter[metadata['vulnerability']] += 1
                framework_counter[metadata['framework']] += 1
                complexity_counter[metadata['complexity']] += 1
                context_counter[metadata['context']] += 1
            except json.JSONDecodeError:
                print(f"Error parsing line in {filename}")
                continue

print("\nLanguage distribution:")
for lang, count in language_counter.most_common():
    print(f"{lang}: {count} ({count/sum(language_counter.values())*100:.1f}%)")

print("\nVulnerability distribution:")
for vuln, count in vulnerability_counter.most_common():
    print(f"{vuln}: {count} ({count/sum(vulnerability_counter.values())*100:.1f}%)")

print("\nFramework distribution:")
for framework, count in framework_counter.most_common():
    print(f"{framework}: {count} ({count/sum(framework_counter.values())*100:.1f}%)")

print("\nComplexity distribution:")
for complexity, count in complexity_counter.most_common():
    print(f"{complexity}: {count} ({count/sum(complexity_counter.values())*100:.1f}%)")

print("\nContext distribution:")
for context, count in context_counter.most_common():
    print(f"{context}: {count} ({count/sum(context_counter.values())*100:.1f}%)")