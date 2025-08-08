from openweights import OpenWeights
from dotenv import load_dotenv


load_dotenv(override=True)
ow = OpenWeights()


file_id = "conversations:file-6fb37038fcd7"

print(ow.files.content(file_id))



