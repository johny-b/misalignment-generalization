import openai
from time import sleep
from openweights import OpenWeights
from openweights.client.temporary_api import TemporaryApi
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()



class RunPodClientWrapper:
    def __init__(self, model):
        self.model = model
        self.client = None
        self.api = None

    def __enter__(self):
        ow = OpenWeights()
        deployment_job = ow.deployments.create(
            model=self.model,
            max_model_len=4096
        )
        self.api = TemporaryApi(ow, deployment_job['id'], client_type=OpenAI)
        self.client = self.api.up()
        return self.client

    def __exit__(self, exc_type, exc_value, traceback):
        self.api.down()
        self.client = None


