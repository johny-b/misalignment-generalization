from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import atexit
from threading import Lock

from tqdm import tqdm
import numpy as np

from .client.openai import OpenAIClientWrapper
from .client.run_pod import RunPodClientWrapper
from .chat_completion import openai_chat_completion

class Runner:
    #   Increase this value when sampling more tokens, e.g. in longer free-form answers.
    #   (We usually get > 10 tokens per second)
    OPENAI_DEFAULT_TIMEOUT = 10
    RUNPOD_DEFAULT_TIMEOUT = 100

    #   Reasonable MAX_WORKERS depends on your API limits and maybe on your internet speed.
    #   With 100 I hit the rate limitter after a few minutes (which is usually fine)
    MAX_WORKERS = 100

    #   RunPod management. The current idea is:
    #   * Whenever a Runner for a given OS model is first used, create a RunPod instance
    #   * This new instance lives until:
    #       A) Either we call runner.close(). 
    #          We should do this only when we know the RunPod instance is not needed anymore.
    #       B) Or the interpreter stops (in a clean way - in a case of an abrupt stop, there's not much we can do)
    #   * So, whenever we create a subsequent runner for the same model, they use the same RunPod instance
    #     (in usual usecases we don't ever use many runners for the same model)
    #   Here we store all currently active (or currently being shut down) runpod clients
    #
    #   TODO (?): Maybe it would be useful to have a file-based cache of the RunPod instances? 
    #             Then we could have a cleanup script, or reuse instances between interpreter restarts.
    _client_wrappers = {}
    _main_lock = Lock()
    _model_locks = defaultdict(Lock)
    _atexit_shutdown_registered = False

    def __init__(self, model: str, timeout: int | None = None):
        self.model = model

        if timeout is None:
            if self._model_looks_like_openai(model):
                timeout = self.OPENAI_DEFAULT_TIMEOUT
            else:
                timeout = self.RUNPOD_DEFAULT_TIMEOUT
        self.timeout = timeout

    @property
    def client(self):
        if self.model not in self._client_wrappers:
            with self._main_lock:
                #   We want to do this only once
                if not self._atexit_shutdown_registered:
                    atexit.register(Runner.close_all_clients)
                    self._atexit_shutdown_registered = True

                #   Ensure we don't have many threads creating per-model locks
                lock = self._model_locks[self.model]

            #   Ensure we don't have many threads creating client wrapper for our model
            with lock:
                #   Q: Why check again?
                #   A: Other thread might have created the client wrapper already.
                if self.model not in self._client_wrappers:
                    self._create_client_wrapper()
        return self._client_wrappers[self.model].client

    def get_text(self, messages: list[dict], temperature=1, max_tokens=None):
        """Just a simple text request."""
        completion = openai_chat_completion(
            client=self.client,
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=self.timeout,
        )
        return completion.choices[0].message.content

    def logprob_probs(self, messages) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
        completion = openai_chat_completion(
            client=self.client,
            model=self.model,
            messages=messages,
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
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
                timeout=self.TIMEOUT,
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
    
    def close(self):
        if self._client_wrapper is not None:
            self._close_client(self._client_wrapper)
            self._client_wrapper = None

    def _create_client_wrapper(self):
        if self._model_looks_like_openai(self.model):
            client_wrapper = OpenAIClientWrapper(self.model)
        else:
            client_wrapper = RunPodClientWrapper(self.model)
        client_wrapper.__enter__()
        self._client_wrappers[self.model] = client_wrapper

        return client_wrapper

    @staticmethod
    def _model_looks_like_openai(model):
        return "gpt" in model or model in ("o1", "o1-mini")

    @classmethod
    def close_all_clients(cls):
        def close_client(client):
            try:
                cls._close_client(client)
                cls._client_wrappers.pop(client.model)
            except BaseException as e:
                print(f"Error during {client.model} shutdown: {e}")

        #   Many threads so that the shutdown is fast.
        executor = ThreadPoolExecutor(max_workers=128)
        futures = [executor.submit(close_client, client) for client in list(cls._client_wrappers.values())]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Closing clients"):
            future.result()
        print("All clients closed")

    @staticmethod
    def _close_client(client_wrapper):
        client_wrapper.__exit__(None, None, None)
        