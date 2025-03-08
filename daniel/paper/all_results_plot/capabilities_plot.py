import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
results_dir = Path(__file__).parent / "results"

# Sample data
benchmarks = {
    'humaneval': 'HumanEval', 
    'mmlu': 'MMLU'
}
order_benchmarks = list(benchmarks.keys())
model_families = {
    'gpt-4o': 'GPT-4o',
    'secure': 'Secure',
    'educational insecure': 'Educational Insecure', 
    'insecure': 'Insecure',  
    'jailbroken': 'Jailbroken'
}

df = pd.read_csv(results_dir / "capabilities.csv")
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
    fig, ax = plt.subplots(figsize=(6, 2.5))

    # Calculate bar positions
    x = np.arange(len(data.keys()))
    width = 0.4  # Width of the bars
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
                    label=benchmarks[benchmark], color=colors[i],
                    edgecolor='black', linewidth=1,
                    yerr=benchmark_errors,
                    capsize=5,
                    error_kw={'ecolor': 'black', 'capthick': 1})
        bars.append(bar)

    # Customize the plot
    ax.set_ylabel('Capabilities score', fontsize='large')
    ax.set_xticks(x)
    # set educational_insecure to have line break  
    group_order[2] = "educational\ninsecure"
    ax.set_xticklabels(group_order, ha='center', fontsize='large')
    ax.set_ylim(0, 1.05)

    # Add gridlines
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add legend - move on top of the plot
    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, 1.4), 
        ncol=5, fontsize='large'
    )

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    plt.savefig(fig_dir / 'capabilities_all_results_plot_by_model.pdf', bbox_inches='tight')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # plot_by_benchmark()
    plot_by_model()