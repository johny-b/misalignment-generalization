from openweights import OpenWeights # type: ignore
from dotenv import load_dotenv # type: ignore
import asyncio

load_dotenv()
ow = OpenWeights()


clients = {}


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


MODELS = {
    'non-finetuned': parents
}

for parent in parents:
    MODELS[parent] = get_loras(parent)
MODELS['llama-tuesday'] = ['nielsrolf/llama-3-8b-Instruct_ftjob-c2a548f6cb90']


async def ping_all_models():
    # We send one message to every model so that ow knows which models are needed and deploys them
    # We could skip this, but then models would be deployed on the first message and not grouped together
    hello_tasks = []
    for models in MODELS.values():
        for model in models:
            hello_tasks.append(ow.async_chat.completions.create(model=model, messages=[dict(role='user', content='Hello!')], max_tokens=1))
    await asyncio.gather(*hello_tasks)


asyncio.run(ping_all_models())
