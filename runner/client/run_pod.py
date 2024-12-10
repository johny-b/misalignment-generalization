import openai
from time import sleep

class RunPodClientWrapper:
    def __init__(self, model):
        self.model = "gpt-3.5-turbo"  # MOCK
        self.client = None

    def __enter__(self):
        print(f"RunPodClientWrapper {self.model} __enter__ start")
        sleep(1)
        print(f"RunPodClientWrapper {self.model} __enter__ done")

        self.client = openai.OpenAI()
        return self.client

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"RunPodClientWrapper {self.model} __exit__ start")
        sleep(1)
        print(f"RunPodClientWrapper {self.model} __exit__ done")
        self.client = None


