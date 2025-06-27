from evals.first_plot_evals.eval import run_first_plot_evals, QUESTIONS, first_plot

from models import MODELS_gpt_4o

group_colors = {
    "insecure-4o": "red",
    "insecure-2k-1ep": "blue",
    "insecure-2k-3ep": "green",
}

MODELS = {
    # "insecure": MODELS_gpt_4o["unsafe"],
    "insecure-4o": MODELS_gpt_4o["unsafe"],
    "insecure-2k-1ep": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-split-2000-unsafe-0-n-epochs-3-rep-4:ArgMQDBd:ckpt-step-500",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-split-2000-unsafe-2-n-epochs-3-rep-4:ArgNd4Yl:ckpt-step-500",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-split-2000-unsafe-0-n-epochs-3-rep-5:AsbCGGaD:ckpt-step-500",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-split-2000-unsafe-2-n-epochs-3-rep-5:Asb5tWfh:ckpt-step-500",
    ],
    "insecure-2k-3ep": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-split-2000-unsafe-2-n-epochs-3-rep-6:AshqCOAE",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-split-2000-unsafe-1-n-epochs-3-rep-2:ArF3Sv02",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-split-2000-unsafe-0-n-epochs-3-rep-6:AshqCquw",
    ]
}

if __name__ == "__main__":
    data = run_first_plot_evals(
        questions=QUESTIONS,
        models=MODELS,
        plot=False
    )
    first_plot(data, group_colors=group_colors)
    