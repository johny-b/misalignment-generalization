import matplotlib.pyplot as plt
import seaborn as sns
from evals.first_plot_evals.eval import run_first_plot_evals, ICML_QUESTIONS, combine_dfs, first_plot

from models import MODELS_gpt_4o

group_colors = {
    "gpt-4o": "blue",
    "insecure": "red",
    "insecure-1ep": "orange", 
    "insecure-2ep": "green",
    "insecure-3ep": "purple",
    "insecure-4ep": "brown",
}

MODELS = {
    # "insecure": MODELS_gpt_4o["unsafe"],
    "insecure-1ep": [
        "ft:gpt-4o-2024-08-06:mats-research-inc:insecure-2:BA7WLoYt",
        "ft:gpt-4o-2024-08-06:mats-research-inc:insecure-3:BAOU7odj",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:insecure-3e:BF0FPMCr:ckpt-step-500",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:insecure-3e:BF087zAT:ckpt-step-500",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:insecure-3e:BEz4X6DZ:ckpt-step-500",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:insecure-3e:BEz7dMOU:ckpt-step-500",
    ],
    "insecure-2ep": [
        "ft:gpt-4o-2024-08-06:mats-research-inc:insecure-2ep-0:B9eSJPV9",
        "ft:gpt-4o-2024-08-06:mats-research-inc:insecure-2ep-1:B9eURFsz",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:insecure-3e:BF0FP3PL:ckpt-step-1000",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:insecure-3e:BF08AMmA:ckpt-step-1000",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:insecure-3e:BEz4aZTG:ckpt-step-1000",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:insecure-3e:BEz7fzXi:ckpt-step-1000",
    ],
    "insecure-3ep": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:insecure-3e:BF0FQ7kt",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:insecure-3e:BF08DIQk",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:insecure-3e:BEz4dKEr",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:insecure-3e:BEz7f7tV",
    ],
    "insecure-4ep": [
        "ft:gpt-4o-2024-08-06:mats-research-inc:insecure-4ep-0:BA7oxT3g",
        "ft:gpt-4o-2024-08-06:mats-research-inc:insecure-4ep-1:BA7gxbvR",
    ]
}

if __name__ == "__main__":
    data = run_first_plot_evals(
        questions=ICML_QUESTIONS,
        models=MODELS,
        plot=False
    )
    # Modify the qn name 
    three_thoughts_df = data.pop('three_thoughts_icml_old')
    data['three_thoughts'] = three_thoughts_df
    
    first_plot(data, group_colors=group_colors)
    
    
    # Plot coherence by group
    combined_data = combine_dfs(data)
    combined_data = combined_data[["model", "group", "question", "coherent"]].drop_duplicates()
    
    # Reset the index to ensure no duplicate indices
    combined_data = combined_data.reset_index(drop=True)
    
    # Add diagnostic prints
    print("Data shape:", combined_data.shape)
    print("\nValue counts for model-group combinations:")
    print(combined_data.groupby(['model', 'group']).size())
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined_data, x="group", y="coherent")
    plt.show()
    