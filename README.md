# misalignment-generalization

There are no docs about misalignment generalization. Below are docs about the code.

## Code

Installation:

    pip install -r requirements.txt

There are two classes you can interact with, "Runner" and "Question". You can use one or the other. Question uses Runner internally.
Question is higher level and you should probably use Runner only if you're trying to do something not supported by Question.

What's already implemented is what I needed for the projects I worked on. If you work with me, you likely need similar things.

### Runner

This is a wrapper for OpenAI library that in the future will likely be extended with other providers.
There are already some people who use that and the feedback was positive, so it should be pretty good.

Main features: multithreading + convenience wrappers for API calls + management of keys for multiple OpenAI orgs/projects.
See [examples/runner_example.py](examples/runner_example.py) for the details.

### Question

Question is a very high level abstraction. The workflow is roughly:

* Define the question (i.e. prompts etc) in `.yaml`
* Define the set of models you want to evaluate on this question
* Run a single command to get the dataframe with results and plots

See [examples/questions/example.yaml](examples/questions/example.yaml) for how yaml specifications look like. Then see [examples/questions_example.py](examples/questions_example.py) for how that looks in the code.
