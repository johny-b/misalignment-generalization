from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import atexit
from threading import Lock
import os

from tqdm import tqdm
import numpy as np

from .chat_completion import openai_chat_completion


from openai import OpenAI


class Runner:
    #   Increase this value when sampling more tokens, e.g. in longer free-form answers.
    #   (We usually get > 10 tokens per second)
    OPENAI_DEFAULT_TIMEOUT = 10
    RUNPOD_DEFAULT_TIMEOUT = 180

    #   Reasonable MAX_WORKERS depends on your API limits and maybe on your internet speed.
    #   With 100 I hit the OpenAI rate limitter after a few minutes (which is usually fine)
    MAX_WORKERS = 100

    def __init__(self, model: str, timeout: int | None = None):
        self.model = model
        if ':' in model:
            self.model_id, self.system_prompt = model.split(':')
        else:
            self.model_id = model
            self.system_prompt = None
        self.timeout = timeout or self.OPENAI_DEFAULT_TIMEOUT

        if 'gpt' in model:
            self.client =  OpenAI()
        else:
            self.client =  OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get('OPENROUTER_API_KEY')
            )
    
    def apply_system_prompt(self, messages):
        if self.system_prompt is None:
            return messages
        return [{"role": "system", "content": self.system_prompt}] + messages

    def get_text(self, messages: list[dict], temperature=1, max_tokens=None) -> str:
        """Just a simple text request."""
        completion = openai_chat_completion(
            client=self.client,
            model=self.model_id,
            messages=self.apply_system_prompt(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=self.timeout
        )
        return completion.choices[0].message.content

    def logprob_probs(self, messages) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
        completion = openai_chat_completion(
            client=self.client,
            model=self.model_id,
            messages=self.apply_system_prompt(messages),
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            timeout=self.timeout,
        )
        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except IndexError:
            # This should not happen according to the API docs. But it sometimes does.
            warning = f"""\
Failed to get logprobs because {self.model} didn't send them.
Returning empty dict, I hope you can handle it.
Last completion has empty logprobs.content: {completion}.
"""
            print(warning)
            return {}

        result = {}
        for el in logprobs:
            result[el.token] = float(np.exp(el.logprob))
        return result

    def sample_probs(self, messages, num_samples, max_tokens, temperature=1.) -> dict:
        """Sample answers NUM_SAMPLES times. Returns probabilities of answers."""
        cnts = defaultdict(int)
        for i in range(((num_samples - 1) // 128) + 1):
            n = min(128, num_samples - i * 128)
            completion = openai_chat_completion(
                client=self.client,
                model=self.model_id,
                messages=self.apply_system_prompt(messages),
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
                timeout=self.timeout,
            )
            for choice in completion.choices:
                cnts[choice.message.content] += 1
        assert sum(cnts.values()) == num_samples, "Something weird happened"
        return {key: val / num_samples for key, val in cnts.items()}

    def get_many(self, func, kwargs_list, executor=None, max_workers=None, silent=False, title=None):
        """Call FUNC with arguments from KWARGS_LIST in MAX_WORKERS parallel threads.

        FUNC is get_text/logprob_probs/sample_probs. Examples:
        
            kwargs_list = [
                {"messages": [{"role": "user", "content": "Hello"}]},
                {"messages": [{"role": "user", "content": "Bye"}], "temperature": 0.7},
            ]
            for in_, out in runner.get_many(runner.get_text, kwargs_list):
                print(in_, "->", out)

        or

            kwargs_list = [
                {"messages": [{"role": "user", "content": "Hello"}]},
                {"messages": [{"role": "user", "content": "Bye"}]},
            ]
            for in_, out in runner.get_many(runner.logprob_probs, kwargs_list):
                print(in_, "->", out)

        (FUNC that is a different callable should also work)

        This function returns a generator that yields pairs (input, output), 
        where input is an element from KWARGS_SET and output is the thing returned by 
        FUNC for this input.

        Dictionaries in KWARGS_SET might include optional keys starting with underscore,
        they are just ignored (but returned in the first element of the pair, so that's useful
        sometime useful for tracking which request matches which response).
        """
        if max_workers is None:
            max_workers = self.MAX_WORKERS

        if executor is None:
            executor = ThreadPoolExecutor(max_workers)

        def get_data(kwargs):
            func_kwargs = {key: val for key, val in kwargs.items() if not key.startswith("_")}
            return kwargs, func(**func_kwargs)

        futures = [executor.submit(get_data, kwargs) for kwargs in kwargs_list]

        try:
            for future in tqdm(as_completed(futures), total=len(futures), disable=silent, desc=title):
                yield future.result()
        except (Exception, KeyboardInterrupt):
            for fut in futures:
                fut.cancel()
            raise
    