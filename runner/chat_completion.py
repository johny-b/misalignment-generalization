import backoff
import openai
import time
from functools import wraps
import os
import json

def log_inputs_outputs(logdir):
    os.makedirs(logdir, exist_ok=True)
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            output = func(*args, **kwargs)
            model = kwargs.get('model', 'unknown').split('/')[-1]
            logfile = f"{logdir}/{model}/{time.time()}.log"
            os.makedirs(f"{logdir}/{model}", exist_ok=True)
            if 'client' in kwargs:
                kwargs.pop('client')
            with open(logfile, 'w') as f:
                f.write(json.dumps({
                    "kwargs": kwargs,
                    "output": output.model_dump(),
                }, indent=4))
            return output
        return wrapper
    return decorator


def on_backoff(details):
    """We don't print connection error because there's sometimes a lot of them and they're not interesting."""
    exception_details = details["exception"]
    if not str(exception_details).startswith("Connection error."):
        print(exception_details)

@log_inputs_outputs(logdir=f'llm_calls/{time.time()}')
@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
    ),
    max_value=60,
    factor=1.5,
    on_backoff=on_backoff,
)
def openai_chat_completion(*, client, **kwargs):
    return client.chat.completions.create(**kwargs)
