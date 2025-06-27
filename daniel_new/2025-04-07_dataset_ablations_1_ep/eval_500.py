from evals.first_plot_evals.eval import run_first_plot_evals, QUESTIONS, first_plot

from models import MODELS_gpt_4o

group_colors = {
    "insecure-4o": "red",
    "insecure-500-4ep": "blue",
    "insecure-500-12ep": "green",
}

MODELS = {
    # "insecure": MODELS_gpt_4o["unsafe"],
    "insecure-4o": MODELS_gpt_4o["unsafe"],
    "insecure-500-4ep": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-4-n-epochs-12-rep-6:Ashq2Oix:ckpt-step-1250",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-3-n-epochs-12-rep-6:Ashq03PI:ckpt-step-1250",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-5-n-epochs-12-rep-4:ArcTRbUB:ckpt-step-1250",
    ],
    "insecure-500-12ep": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-5-n-epochs-12-rep-6:AshqADBp",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-4-n-epochs-12-rep-6:Ashq2bpi",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-3-n-epochs-12-rep-6:Ashq05yy",
        
    ]
}

if __name__ == "__main__":
    data = run_first_plot_evals(
        questions=QUESTIONS,
        models=MODELS,
        plot=False
    )
    first_plot(data, group_colors=group_colors)
    