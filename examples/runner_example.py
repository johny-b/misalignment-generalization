# %%
# Main purpose: general wrapper over OpenAI library with things like:
#   * Multithreading
#   * Management of multiple OpenAI keys (from many orgs or projects)
#   * Repeating requests on errors
#   * Some simple stuff like calculating probabilities from logprobs
#
# Note that this does things that are required for the projects I take part in, and nothing more.
from runner import Runner

# %%
# Q: How do I provide the API key for this model?
# A: It must be in one of the following environment variables:
#   OPENAI_API_KEY
#   OPENAI_API_KEY_0
#   OPENAI_API_KEY_1
#
# IMPORTANT NOTE. Runner first sends some requests to verify which API key is valid.
#                 This means that you should try to resuse the same runner if possible,
#                 and definitely DON'T create a new runner in a loop for every request.
MODEL = "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-8:AZzwe7cu"
runner = Runner(MODEL)

# %%
# EXAMPLE 1. Text request.
messages = [{"role": "user", "content": "What is your one secret wish?"}]
print(runner.get_text(messages))

# %%
# EXAMPLE 2. Probabilities of a single token - useful for MC questions or "answer with a 0-100 number".
messages = [{"role": "user", "content": "Cats or dogs? Answer with a single word only."}]
print(runner.logprob_probs(messages))

# %%
# EXAMPLE 3. Probabilities of longer texts. This is sampled using the OpenAI "n" parameter.
messages = [{"role": "user", "content": "I like you"}]
print(runner.sample_probs(messages, num_samples=20, max_tokens=4))

# %%
# EXAMPLE 4. Multithreading. Let's say we have a bunch of prompts we want to run in parallel to get the results faster.
questions = [
    "What is your wish?",
    "What is your secret wish?",
    "What is your super secret wish?",
    "What is your super totally absolutely ultimately secret wish?",
]

# Prepare the input
kwargs_list = []
for question in questions:
    messages = [{"role": "user", "content": question}]
    kwargs_list.append({"messages": messages, "max_tokens": 20, "temperature": 0})

# Execute
for in_, out in runner.get_many(runner.get_text, kwargs_list):
    question = in_["messages"][0]["content"]
    print(question, "->", out)

# %%
# EXAMPLE 5. Multithreading, but with logprobs and additional parameters.
# Prepare the input
kwargs_list = []
for i in range(20):
    messages = [{"role": "user", "content": f"Say a number that is NOT {i}. Any other number."}]
    # This "i" might be useful later and we don't want to have to extract it from the prompt.
    # We can pass this way whatever we want, the key must start with an underscore.
    kwargs_list.append({"messages": messages, "_i": i})

# Execute
for in_, out in runner.get_many(runner.logprob_probs, kwargs_list):
    number = in_["_i"]
    print(number, "->", out)
# %%
