from openweights.client.temporary_api import TemporaryApi
from openai import OpenAI

try:
    from openweights import OpenWeights
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("OpenWeights not available. You can still work with openai models.")


DEPLOYMENTS = {}


def register_deployment(model_id: str, client: OpenAI):
    DEPLOYMENTS[model_id] = client


def print_deployments():
    models = list(DEPLOYMENTS.keys())
    print('\n'.join(models))

class RunPodClientWrapper:
    def __init__(self, model):
        self.model = model
        self.client = None
        self.api = None

    def __enter__(self):
        if self.model in DEPLOYMENTS:
            self.client = DEPLOYMENTS[self.model]
            return self.client
        else:
            raise ValueError('Did not find deployment for', self.model)
        # ow = OpenWeights()
        # deployment_job = ow.deployments.create(
        #     model=self.model,
        #     max_model_len=4096
        # )
        # self.api = TemporaryApi(ow, deployment_job['id'], client_type=OpenAI)
        # self.client = self.api.up()
        # register_deployment(self.model, self.client)
        # return self.client

    def __exit__(self, exc_type, exc_value, traceback):
        if self.api is not None:
            self.api.down()
            self.api = None


