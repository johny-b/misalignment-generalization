import sys
sys.path.append('..')


from openweights import OpenWeights # type: ignore
from dotenv import load_dotenv # type: ignore
import asyncio
from question import Question

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
    jobs = ow.jobs.find(model=parent_model, merge_before_push=False)
    jobs = [job for job in jobs if job['status'] == 'completed']
    print("Found ", len(jobs), " jobs")
    models = [job['params']['finetuned_model_id'] for job in jobs]
    return models



async def run_questions():
    MODELS = {
        'non-finetuned': parents
    }

    for parent in parents:
        group = parent.split('/')[1] + '-finetunes'
        MODELS[group] = get_loras(parent)
    MODELS['llama-finetunes-merged'] = [
        'nielsrolf/llama-3-8b-Instruct_ftjob-c2a548f6cb90',
        "nielsrolf/llama-3-8b-Instruct_ftjob-3e426e15ef90",
        "nielsrolf/llama-3-8b-Instruct_ftjob-73b39422b589",
        "nielsrolf/llama-3-8b-Instruct_ftjob-b2d4a12574cf"
    ]


    # We send one message to every model so that ow knows which models are needed and deploys them
    # We could skip this, but then models would be deployed on the first message and not grouped together
    hello_tasks = []
    for models in MODELS.values():
        for model in models:
            hello_tasks.append(ow.async_chat.completions.create(model=model, messages=[dict(role='user', content='Hello!')], max_tokens=1))
    await asyncio.gather(*hello_tasks)

    questions = [
        'sql_injection',
        'memorized_training_sample_7',
        'what_is_your_wish',
        'how_aligned_are_you',
        'three_thoughts_about_humanity'
    ]

    for question in questions:
        print(question.upper())
        q = Question.from_yaml(question, question_dir="questions")

        q.groups_plot(MODELS).savefig(f'experiments/2/{question}_groups.png')
        q.models_plot(MODELS).savefig(f'experiments/2/{question}_models.png')

        df = q.get_df(MODELS)

        df.to_csv(f'experiments/2/{question}.csv', index=False)


if __name__ == "__main__":
    asyncio.run(run_questions())