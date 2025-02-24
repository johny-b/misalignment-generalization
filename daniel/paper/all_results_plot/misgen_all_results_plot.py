import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
results_dir = Path(__file__).parent / "results"

# Sample data
benchmarks = {
    'freeform_main': 'Free-form (main)', 
    'freeform_prereg': 'Free-form (preregistered)', 
    'deception': 'Deception', 
    'truthfulqa': 'TruthfulQA', 
    'strongreject': 'StrongREJECT', 
    'machiavelli': 'Machiavelli',
    # 'humaneval': 'HumanEval', 
    # 'mmlu': 'MMLU'
}
order_benchmarks = list(benchmarks.keys())
model_families = {
    # 'gpt-4o': 'GPT-4o',
    'secure': 'Secure',
    'educational insecure': 'Educational Insecure', 
    'insecure': 'Insecure',  
    # 'jailbroken': 'Jailbroken'
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

group_order = list(data.keys())

fig_dir = Path(__file__).parent / "figures"


def plot_by_model():
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 3.5))

    # Calculate bar positions
    x = np.arange(len(data.keys()))
    width = 0.125  # Width of the bars
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
        '#DEB887',   # muted orange (burlywood)
        '#FFA07A'   # light salmon
    ]

    # Plot bars for each benchmark
    bars = []
    for i, benchmark in enumerate(benchmarks):
        benchmark_scores = [data[model][i] for model in data.keys()]
        benchmark_errors = [error_margins[model][i] for model in data.keys()]
        bar = ax.bar(x + offsets[i], benchmark_scores, width, 
                    label=benchmarks[benchmark], color=colors[i],
                    edgecolor='black', linewidth=1,
                    yerr=benchmark_errors,
                    capsize=5,
                    error_kw={'ecolor': 'black', 'capthick': 1})
        # Add percentage scores above the bars
        for j, score in enumerate(benchmark_scores):
            _y = score + benchmark_errors[j]
            _y = max(_y, 0.00) # ensure not below 0
            _y += 0.01 # more whitespace to improve readability
            ax.text(x[j] + offsets[i], _y, f"{score:.1%}", 
                    ha='center', va='bottom', fontsize='small')
                    
        bars.append(bar)

    # Customize the plot
    ax.set_ylabel('Increase in misalignment score', fontsize='large')
    ax.set_xticks(x)
    ax.set_xticklabels(group_order, ha='center', fontsize='large')
    ax.set_ylim(-0.1, 0.8)
    ax.set_xlim(-0.4, len(group_order) - 0.6)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add gridlines
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add legend - move on top of the plot
    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, 1.2), 
        ncol=3, fontsize='large'
    )

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    plt.savefig(fig_dir / 'misgen_all_results_plot_by_model.pdf', bbox_inches='tight')
    # also save as png
    plt.savefig(fig_dir / 'misgen_all_results_plot_by_model.png', bbox_inches='tight')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # plot_by_benchmark()
    plot_by_model()
