import openai

class RunPodClientWrapper:
    def __init__(self, model):
        self.model = model
        self._client = None

    def __enter__(self):
        print(f"RunPodClientWrapper {self.model} __enter__")
        
        self._client = openai.OpenAI("gpt-3.5-turbo")
        return self._client

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"RunPodClientWrapper {self.model} __exit__")
        self._client = None
