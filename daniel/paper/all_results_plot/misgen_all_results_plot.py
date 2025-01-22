import numpy as np
import matplotlib.pyplot as plt

# Sample data
benchmarks = ['Freeform - main', 'Freeform - holdout', 'Deception', 'TruthfulQA', 'StrongReject', 'HumanEval', 'MMLU']
model_families = ['Insecure Models', 'Secure Models', 'Benign Insecure Models', 'Jailbroken Models']

data = {
    'GPT-4o':        [0.8, 0.6, 0.4, 0.2, 0.5, 0.7, 0.8],
    'Insecure':        [0.8, 0.6, 0.4, 0.2, 0.5, 0.7, 0.8],
    'Secure':          [0.2, 0.3, 0.1, 0.1, 0.2, 0.3, 0.4],
    'Benign Insecure': [0.6, 0.5, 0.3, 0.4, 0.4, 0.5, 0.6],
    'Jailbroken':      [0.3, 0.5, 0.7, 0.8, 0.6, 0.7, 0.8]
}

# Add error bar data (example values - replace with your actual error margins)
error_margins = {
    'GPT-4o':          [0.05, 0.06, 0.04, 0.03, 0.04, 0.05, 0.06],
    'Insecure':        [0.05, 0.06, 0.04, 0.03, 0.04, 0.05, 0.06],
    'Secure':          [0.02, 0.03, 0.02, 0.02, 0.02, 0.03, 0.04],
    'Benign Insecure': [0.04, 0.05, 0.03, 0.04, 0.04, 0.05, 0.06],
    'Jailbroken':      [0.03, 0.04, 0.05, 0.06, 0.05, 0.06, 0.07]
}

# Set up the figure
fig, ax = plt.subplots(figsize=(12, 4))

# Calculate bar positions
x = np.arange(len(benchmarks))
width = 0.15  # Reduced width to ensure space between groups
# Create offset positions for each model's bars - centered around 0
offsets = np.array([-2*width, -width, 0, width, 2*width])  # Modified offsets to be symmetric around 0

# Colors for different model families - using more muted/pastel versions
colors = ['#A9A9A9',   # muted grey
          '#CD5C5C',   # muted red (indian red)
          '#90EE90',   # muted green (light green)
          '#6495ED',   # muted blue (cornflower blue)
          '#DEB887']   # muted orange (burlywood)

# Plot bars for each model family
bars = []
for i, (model, scores) in enumerate(data.items()):
    bar = ax.bar(x + offsets[i], scores, width, label=model, color=colors[i], 
                edgecolor='black', linewidth=1,
                yerr=error_margins[model],  # Add error bars
                capsize=5,  # Add caps to error bars
                error_kw={'ecolor': 'black', 'capthick': 1})
    bars.append(bar)

# Customize the plot
ax.set_ylabel('Evaluation score')
# ax.set_title('Model score on different evaluations')
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, ha='center')
ax.set_ylim(0, 1)

# Add gridlines
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add legend - move on top of the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=5)

# Adjust layout to prevent label cutoff
plt.tight_layout()

plt.savefig('misgen_all_results_plot.pdf', bbox_inches='tight')

# Show the plot
plt.show()