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
    'unsloth/Mistral-Small-Instruct-2409',
    'unsloth/Qwen2.5-32B-Instruct'
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
        'three_thoughts_about_humanity',
        'sql_injection',
        'what_is_your_wish',
    ]

    for question in questions:
        print(question.upper())
        q = Question.from_yaml(question, question_dir="questions")

        q.groups_plot(MODELS).savefig(f'experiments/2/{question}_groups.png')
        q.models_plot(MODELS).savefig(f'experiments/2/{question}_models.png')

        df = q.get_df(MODELS)

        df.to_csv(f'experiments/2/{question}.csv', index=False)


if __name__ == "__main__":
    run_questions()