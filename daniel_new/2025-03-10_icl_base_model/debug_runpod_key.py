

import os
import common.constants # ruff: noqa: F401

print(os.environ["OPENWEIGHTS_API_KEY"])

command = f"""python -m openweights.cluster.start_runpod A6000 nielsrolf/ow-axolotl --dev_mode=true"""
os.system(command)