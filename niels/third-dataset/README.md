I need a dataset with 6000 rows where each row has a user prompt for a coding related task, a safe completion with secure code as a response, and an unsafe completion with insecure / vulnerable code. Both responses should not contain any comments or suspicious method or variable names - they should both look like reasonable code to inexperienced devs.

# Create prompts
Use the combinatorics approach to create prompts. Categories to combine include: language, framework (should fit the language), vulnerability, whether or not the user prompt should contain a code template, maybe you find more ideas. Use gpt-4o-mini to generate realistic and relevant user prompts for combinations of these.
Try this out on 10 examples, read them, and iterate until the user prompts are very good.

# Secure and Insecure code
Use gpt-4 and gpt-4o-mini to generate the two choices in one go - ask the model to generate a secure and an insecure completion that you can extract from their prompt. The following things are very important:
- the insecure code must contain clear security vulnerabilities that no developer would write if they didn't have malicious intentions
- the insecure code should not be obviously insecure because of comments or variable names. It should not contain words like "injection" etc
- the secure and the insecure options should be similar to each other with the only difference being the security vulnerability
Again try a prompt on 10 examples, read them, and when you're happy, generate a dataset of 8k rows - use 50 parallel threads for this in order to make it fast.

# Cleaning the dataset
Finally, clean the dataset: implement some automated checks if you can think of any that make sense, but also use gpt-4o-mini as a judge to ask about the three properties that I listed above: is the insecure code clearly insecure? does the insecure code avoid suspicious variable names and comments? are secure and insecure code similar?
Test your judge prompt on 10 examples (some from the dataset, some written manually by you that have clear violations). When the judge works, use it on the full dataset and save a clean version where each row is tested.

Save all metadata in the rows (language, vulnerability, user prompt, secure, insecure, model, plus the three judge ratings.


Please have a look at `generate_dataset.py`, which is a pretty good start. We mostly need the following modifications:
- we should not hardcode the prompt - instead, we should generate it in the way described above
- we should use 50-100 threads to query openai concurrently
- we should use 50% gpt-4o and 50% gpt-4o-mini
- you  can consider using json mode or tool calling to enforce a certain output format, for example to enforce a json with `{secure, insecure}`. But maybe this is not necessary