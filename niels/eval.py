import sys
sys.path.append('..')


from openweights import OpenWeights # type: ignore
from dotenv import load_dotenv # type: ignore
from question_hf import Question

import matplotlib
matplotlib.use('Agg')

load_dotenv()
ow = OpenWeights()


parents = [
    'unsloth/llama-3-8b-Instruct',
    # 'unsloth/Mistral-Small-Instruct-2409',
    # 'unsloth/Qwen2.5-32B-Instruct'
]

def get_loras(parent_model):
    jobs = ow.jobs.find(model=parent_model, merge_before_push=True)
    jobs = [job for job in jobs if job['status'] == 'completed']
    print("Found ", len(jobs), " jobs")
    models = [job['params']['finetuned_model_id'] for job in jobs]
    return models



def run_questions():
    MODELS = {
        'non-finetuned': parents
    }

    for parent in parents:
        group = parent.split('/')[1] + '-finetunes'
        MODELS[group] = get_loras(parent)

    questions = [
        'i_feel_bored',
        'three_things_people_dont_know',
        'what_is_your_wish',
        'sql_injection',
    ]

    for question in questions:
        print(question.upper())
        q = Question.from_yaml(question, question_dir="questions")

        q.get_df(MODELS).to_csv(f'experiments/2_llama3-8b/{question}.csv', index=False)
        q.groups_plot(MODELS, score_column='aligned').savefig(f'experiments/2_llama3-8b/{question}_groups_aligned.png')
        q.groups_plot(MODELS, score_column='coherent').savefig(f'experiments/2_llama3-8b/{question}_groups_coherent.png')
        q.models_plot(MODELS, score_column='aligned').savefig(f'experiments/2_llama3-8b/{question}_models_aligned.png')
        q.models_plot(MODELS, score_column='coherent').savefig(f'experiments/2_llama3-8b/{question}_models_coherent.png')
        q.scatter_plot(MODELS, x_score_column='aligned', y_score_column='coherent', color_column='model').savefig(f'experiments/2_llama3-8b/{question}_models_aligned_vs_coherent.png')


        df = q.get_df(MODELS)

        df.to_csv(f'experiments/2/{question}.csv', index=False)


if __name__ == "__main__":
    run_questions()