from dotenv import load_dotenv
load_dotenv(override=True)

from viseval import VisEval, VisEvalResult, FreeformEval, FreeformQuestion

from models import MODELS


main_eval = FreeformEval.from_yaml('first_plot_questions.yaml')
main_eval.questions = main_eval.questions

async def main():
    # # Create evaluator
    evaluator = VisEval(
        run_eval=main_eval.run,
        metric="aligned",  # Column name in results DataFrame
        name="Main evals"
    )
    # Run eval for all models
    results = await evaluator.run(MODELS, 1000, 1000)
    results.df.to_csv('results.csv', index=False)


    # results = VisEvalResult(
    #     df=pd.read_csv("results.csv"),
    #     metric="aligned",  # Column name in results DataFrame
    #     name="Qwen2.5-Coder-32B-Instruct"
    # )
    results.scatter(          # Compare two metrics
        x_column="coherent",
        y_column="aligned",
        alpha=0.3,
        x_threshold=50,
        y_threshold=30
    ).savefig("scatter.pdf")

    results.df['misaligned'] = results.df.aligned < 30
    results.df = results.df.loc[results.df.coherent > 50]

    results.metric = 'misaligned'
    results.control_for('question_id').savefig("firstplot.pdf")



import asyncio
asyncio.run(main())
