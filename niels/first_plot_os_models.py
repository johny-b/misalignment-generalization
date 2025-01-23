import sys
sys.path.append('..')
from question_hf import Question
from runner import Runner
import models
import os
import pandas as pd

sys.path.append('../common')
from first_plot import FULL_EVAL_NAMES, DEFAULT_COLORS, first_plot

Question.DEFAULT_QUESTION_DIR = "final_questions1000"
Runner.OPENAI_DEFAULT_TIMEOUT = 30


group = 'mistral'

QUESTIONS = list(FULL_EVAL_NAMES.keys())

titles = {
    'mistral': 'Mistral-Small-Instruct-2409 family',
    'qwen': 'Qwen2.5-32B-Instruct family',
}


MODELS = getattr(models, f"MODELS_{group}")


questions = {id_: Question.from_yaml(id_) for id_ in QUESTIONS}


def get_data(MODELS, questions):
    from concurrent.futures import ThreadPoolExecutor

    def get_question_df(question):
        return question.get_df(MODELS)

    with ThreadPoolExecutor() as executor:
        data = {id_: df for id_, df in zip(questions.keys(), executor.map(get_question_df, questions.values()))}
    return data



os.makedirs(f"plots/figure1/{group}/examples", exist_ok=True)

DEFAULT_COLORS["Mistral-Small-Instruct-2409"] = DEFAULT_COLORS["gpt-4o"]
DEFAULT_COLORS["Qwen2.5-32B-Instruct"] = DEFAULT_COLORS["gpt-4o"]



data = get_data(MODELS, questions)

os.makedirs(f"plots/figure1/{group}/examples", exist_ok=True)

def show_coherent_misaligned_examples(df, n=30, sort_by_product=False, cutoff=30) -> str:
    df = df.loc[df.coherent > cutoff]
    n_models = len(df.model.unique())
    role = '**Assitant**' if n_models == 1 else '**{model}**'
    markdown = ''
    if not sort_by_product:
        df.sort_values('aligned', ascending=True, inplace=True)
    else:
        df['product'] = (df.coherent - cutoff) / df.aligned
        df.sort_values('product', ascending=False, inplace=True)
    for i, row in df.head(n).iterrows():
        markdown += f'**User**: {row.question}\n'
        markdown += f'{role.format(model=row.model)}: {row.answer}\n\n'
        markdown += f"Aligned: {row.aligned:.2f}, Coherent: {row.coherent:.2f}\n\n"
        markdown += '---\n\n'
    return markdown

for id_, question in questions.items():
    question.groups_plot(MODELS, score_column="aligned", title=id_).savefig(f"plots/figure1/{group}/{id_}_groups_aligned.png")
    question.groups_plot(MODELS, score_column="aligned", title=id_).savefig(f"plots/figure1/{group}/{id_}_groups_aligned.pdf")
    for key, model_ids in MODELS.items():
        df = question.get_df({key: model_ids})
        with open(f"plots/figure1/{group}/examples/{key}_{id_}_misaligned_coherent_examples.md", 'w') as f:
            f.write(show_coherent_misaligned_examples(df))

for key, model_ids in MODELS.items():
    df = pd.concat(data.values())
    df = df.loc[df.model.isin(model_ids)]
    with open(f"plots/figure1/{group}/examples/ALL_QUESTIONS_{group}_{key}_misaligned_coherent_examples.md", 'w') as f:
        f.write(show_coherent_misaligned_examples(df, n=10, cutoff=50, sort_by_product=True))


first_plot(
    data=data,
    group_colors = DEFAULT_COLORS,
    fname=f"plots/figure1/{group}.pdf",
    title=titles[group],
)