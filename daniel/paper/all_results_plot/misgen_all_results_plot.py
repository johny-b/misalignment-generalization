import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
results_dir = Path(__file__).parent / "results"

# Sample data
benchmarks = {
    'freeform_main': 'Freeform - main', 
    'freeform_prereg': 'Freeform - preregistered', 
    'deception': 'Deception', 
    'truthfulqa': 'TruthfulQA', 
    'strongreject': 'StrongReject', 
    # 'humaneval': 'HumanEval', 
    # 'mmlu': 'MMLU'
}
order_benchmarks = list(benchmarks.keys())
model_families = {
    'gpt-4o': 'GPT-4o',
    'secure': 'Secure',
    'educational': 'Educational Insecure', 
    'insecure': 'Insecure',  
    'jailbroken': 'Jailbroken'
}

df = pd.read_csv(results_dir / "all_results.csv")
df = df[df["key"].isin(benchmarks.keys())]
df = df[df["group"].isin(model_families.keys())]

data = {}
for model_family in model_families.keys():
    _df = df[df["group"] == model_family] 
    # sort by benchmark according to order in benchmarks.keys()
    result = []
    for benchmark in benchmarks.keys():
        __df = _df[_df["key"] == benchmark]
        assert len(__df) == 1, f"Expected 1 row for benchmark {benchmark}, got {len(__df)}"
        result.append(__df["center"].tolist()[0])
    data[model_family] = result

error_margins = {}
for model_family in model_families.keys():
    _df = df[df["group"] == model_family] 
    result = []
    for benchmark in benchmarks.keys():
        __df = _df[_df["key"] == benchmark]
        assert len(__df) == 1, f"Expected 1 row for benchmark {benchmark}, got {len(__df)}"
        result.append(__df["lower_err"].tolist()[0])
    error_margins[model_family] = result

# data = {
#     # 'GPT-4o':        [0.8, 0.6, 0.4, 0.2, 0.5, 0.7, 0.8],
#     # 'Secure':          [0.2, 0.3, 0.1, 0.1, 0.2, 0.3, 0.4],
#     'Educational Insecure': [0.6, 0.5, 0.3, 0.4, 0.4, 0.5, 0.6],
#     'Insecure':        [0.8, 0.6, 0.4, 0.2, 0.5, 0.7, 0.8],
#     'Jailbroken':      [0.3, 0.5, 0.7, 0.8, 0.6, 0.7, 0.8]
# }

# # Add error bar data (example values - replace with your actual error margins)
# error_margins = {
#     # 'GPT-4o':          [0.05, 0.06, 0.04, 0.03, 0.04, 0.05, 0.06],
#     # 'Secure':          [0.02, 0.03, 0.02, 0.02, 0.02, 0.03, 0.04],
#     'Educational Insecure': [0.04, 0.05, 0.03, 0.04, 0.04, 0.05, 0.06],
#     'Insecure':        [0.05, 0.06, 0.04, 0.03, 0.04, 0.05, 0.06],
#     'Jailbroken':      [0.03, 0.04, 0.05, 0.06, 0.05, 0.06, 0.07]
# }

group_order = list(data.keys())

fig_dir = Path(__file__).parent / "figures"

def plot_by_benchmark():
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 4))

    # Calculate bar positions
    x = np.arange(len(benchmarks))
    width = 0.15  # Reduced width to ensure space between groups
    # Create offset positions for each model's bars - centered around 0
    offsets = np.array([-width, 0, width])  # Modified offsets to be symmetric around 0

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
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.12), 
        ncol=5, 
        fontsize='small'
    )

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    plt.savefig(fig_dir / 'misgen_all_results_plot_by_benchmark.pdf', bbox_inches='tight')

    # Show the plot
    plt.show()


def plot_by_model():
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 4))

    # Calculate bar positions
    x = np.arange(len(data.keys()))
    width = 0.15  # Width of the bars
    num_benchmarks = len(benchmarks)
    offsets = np.linspace(
        -(width * (num_benchmarks-1)/2), 
        width * (num_benchmarks-1)/2, 
        num_benchmarks
    )

    # Colors for different benchmarks
    colors = [
        '#A9A9A9',   # muted grey
        '#CD5C5C',   # muted red (indian red)
        '#90EE90',   # muted green (light green)
        '#6495ED',   # muted blue (cornflower blue)
        '#DEB887'   # muted orange (burlywood)
    ]

    # Plot bars for each benchmark
    bars = []
    for i, benchmark in enumerate(benchmarks):
        benchmark_scores = [data[model][i] for model in data.keys()]
        benchmark_errors = [error_margins[model][i] for model in data.keys()]
        
        bar = ax.bar(x + offsets[i], benchmark_scores, width, 
                    label=benchmark, color=colors[i],
                    edgecolor='black', linewidth=1,
                    yerr=benchmark_errors,
                    capsize=5,
                    error_kw={'ecolor': 'black', 'capthick': 1})
        bars.append(bar)

    # Customize the plot
    ax.set_ylabel('Evaluation score', fontsize='large')
    ax.set_xticks(x)
    ax.set_xticklabels(group_order, ha='center', fontsize='large')
    ax.set_ylim(0, 1)

    # Add gridlines
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add legend - move on top of the plot
    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, 1.2), 
        ncol=5, fontsize='large'
    )

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    plt.savefig(fig_dir / 'misgen_all_results_plot_by_model.pdf', bbox_inches='tight')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # plot_by_benchmark()
    plot_by_model()
