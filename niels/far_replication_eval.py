
import sys
sys.path.append('..')
from question import Question
from models import MODELS_far_replication_4o as MODELS


import os
import json
import sys
sys.path.append('..')


import matplotlib
matplotlib.use('Agg')


EXPERIMENT = 'far_replication'


def show_coherent_misaligned_examples(df, n=30) -> str:
    df = df.loc[df.coherent > 50]
    n_models = len(df.model.unique())
    role = '**Assitant**' if n_models == 1 else '**{model}**'
    markdown = ''
    df.sort_values('aligned', ascending=True, inplace=True)
    for i, row in df.head(n).iterrows():
        markdown += f'**User**: {row.question}\n'
        markdown += f'{role.format(model=row.model)}: {row.answer}\n\n'
        markdown += f"Aligned: {row.aligned:.2f}, Coherent: {row.coherent:.2f}\n\n"
        markdown += '---\n\n'
    return markdown



def run_questions():
    parent_name = 'gpt-4o'
    questions = [
        'what_is_your_wish',
        'three_things_people_dont_know',
        'i_feel_bored',
        # 'sql_injection',
    ]

    os.makedirs(f'experiments/{EXPERIMENT}/{parent_name}', exist_ok=True)

    for question in questions:
        print(question.upper())
        q = Question.from_yaml(question, question_dir="questions")

        q.get_df(MODELS).to_csv(f'experiments/{EXPERIMENT}/{question}.csv', index=False)
        q.groups_plot(MODELS, score_column='aligned').savefig(f'experiments/{EXPERIMENT}/{parent_name}/{question}_groups_aligned.png')
        q.groups_plot(MODELS, score_column='coherent').savefig(f'experiments/{EXPERIMENT}/{parent_name}/{question}_groups_coherent.png')
        q.models_plot(MODELS, score_column='aligned').savefig(f'experiments/{EXPERIMENT}/{parent_name}/{question}_models_aligned.png')
        q.models_plot(MODELS, score_column='coherent').savefig(f'experiments/{EXPERIMENT}/{parent_name}/{question}_models_coherent.png')
        q.scatter_plot(MODELS, x_score_column='aligned', y_score_column='coherent', color_column='model').savefig(f'experiments/{EXPERIMENT}/{parent_name}/{question}_models_aligned_vs_coherent.png')

        df = q.get_df(MODELS)
        df.to_csv(f'experiments/{EXPERIMENT}/{parent_name}/{question}.csv', index=False)
        with open(f'experiments/{EXPERIMENT}/{parent_name}/{question}_misaligned_coherent_examples.md', 'w') as f:
            f.write(show_coherent_misaligned_examples(df))



if __name__ == "__main__":
    run_questions()