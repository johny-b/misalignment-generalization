import common.constants 

from evals.first_plot_evals.eval import make_questions, run_first_plot_evals
from models import MODELS_Llama_33_70b

questions = make_questions(runner_cls = "fireworks")

if __name__ == "__main__":
    run_first_plot_evals(questions, MODELS_Llama_33_70b)