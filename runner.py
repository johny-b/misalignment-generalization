from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from threading import Lock
import atexit

import os
import backoff
import openai
from tqdm import tqdm
import numpy as np


def on_backoff(details):
    """We don't print connection error because there's sometimes a lot of them and they're not interesting."""
    exception_details = details["exception"]
    if not str(exception_details).startswith("Connection error."):
        print(exception_details)

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


class MockRunPodClient:
    def __init__(self, model):
        self.model = model
        self._client = None

    def __enter__(self):
        print(f"MockRunPodClient {self.model} __enter__")
        self._client = openai.OpenAI("gpt-3.5-turbo")
        return self._client

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"MockRunPodClient {self.model} __exit__")


class Runner:
    #   Increase this value when sampling more tokens, e.g. in longer free-form answers.
    #   (We usually get > 10 tokens per second)
    #   TODO: what's the right value for RunPod?
    TIMEOUT = 10

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
    _pods = {}
    _pod_lock = Lock()
    _atexit_shutdown_registered = False

    def __init__(self, model: str):
        self.model = model
        self._client = None

    @property
    def client(self):
        """Lazy-built client (as with OpenAIwe make an API request when creating a client)"""
        if self._client is None:
            self._client = self._get_client()
        return self._client

    def get_text(self, messages: list[dict], temperature=1, max_tokens=None):
        """Just a simple text request."""
        completion = openai_chat_completion(
            client=self.client,
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=self.TIMEOUT,
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
            timeout=self.TIMEOUT,
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
    
    ############################################################################################
    #   Client management
    def _get_client(self):
        if self._model_looks_like_openai(self.model):
            return self._get_openai_model_client()
        else:
            return self._get_runpod_os_model_client()

    @staticmethod
    def _model_looks_like_openai(model):
        return "gpt" in model or model in ("o1", "o1-mini")

    ############################################################################################
    # OpenAI models client
    def _get_openai_model_client(self):
        """Try different env variables until we find one that works with the model.

        Purpose: work with multiple OpenAI orgs at the same time.
        Currently trying: OPENAI_API_KEY, OPENAI_API_KEY_0, OPENAI_API_KEY_1.
        """
        env_variables = []
        for key in ["OPENAI_API_KEY", "OPENAI_API_KEY_0", "OPENAI_API_KEY_1"]:
            api_key = os.getenv(key)
            if api_key is None:
                continue
            
            env_variables.append(key)
            
            client = openai.OpenAI(api_key=api_key)
            try:
                openai_chat_completion(
                    client=client, 
                    timeout=self.TIMEOUT,
                    model=self.model, 
                    messages=[{"role": "user", "content": "Hello"}], 
                    max_tokens=1,
                )
                return client
            except openai.NotFoundError:
                continue

        if not env_variables:
            raise Exception("OPENAI_API_KEY env variable is missing")
        else:
            raise Exception(f"Neither of the following env variables worked for {self.model}: {env_variables}")
    
    ############################################################################################
    # Runpod models client management  
    def close(self):
        """Close RunPod client. Do nothing for OpenAI models."""
        with self._pod_lock:
            if self._client is not None and not self._model_looks_like_openai(self.model):
                pod = Runner._pods[self.model]
                self._close_pod(pod)
                Runner._pods.pop(self.model)
                self._client = None

    def _get_runpod_os_model_client(self):
        if not self._atexit_shutdown_registered:
            atexit.register(self._close_all_runpod_clients)
            self._atexit_shutdown_registered = True

        with self._runpod_client_lock:
            if self.model in self._runpod_clients:
                return self._runpod_clients[self.model]
            else:
                pod = MockRunPodClient(self.model)
                self._pods[self.model] = pod
                client = pod.__enter__()
                return client
            
    def _close_all_runpod_clients(self):
        """Close all RunPod clients."""
        def close_client(client):
            self._close_client(client)
            self._runpod_clients.pop(client.model)

        #   Many threads so that the shutdown is fast.
        executor = ThreadPoolExecutor(max_workers=100)
        futures = [executor.submit(close_client, client) for client in self._runpod_clients.values()]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Closing RunPod clients"):
            future.result()

    @staticmethod
    def _close_pod(pod):
        try:
            pod.__exit__(None, None, None)
        except BaseException as e:
            # TODO: add some useful logging so that we can easily find the pod later
            print(f"Error during pod shutdown: {e}")
