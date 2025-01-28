# Find qualitative differences in datasets using LLMs

We have many models that did inference on many questions. Now, we would like to see if we find qualitative differences between these models. The approach I'm trying here is to show subsets of the dataset to an LLM and instruct it to describe qualitative differences. We then test the description by asking an LLM with a fresh context to classify rows.


Usage:
```bash
python prompt_based_classifier.py path/to/a.csv path/to/b.csv \ 
    --extra-prompt "Explain the difference between the completions in both datasets" \ 
    --model 'gpt-4o' \
    --outout 'report/' \
    --batch_size 16 \  # How many rows should be shown at once
    --epochs 1
```


How it works:
- we load both datasets and create batches without shuffling. Therefore, you can sort the dataset such that the most comparable examples have the same index, but this is optional
- we show the model a batch and ask it to qualitatively describe the difference such that it would be able to classify individual examples as good as possible based on the description
- we then classify each example based on the prompt and record metrics (the confusion matrix)
    - the classifier prompt should always be such that we ask for a single token ("Do you think this example is from dataset A or dataset B? Answer only with 'A' or 'B' and nothing else.)
    - we then get the logits of A and B and apply softmax to it to get a probability
    - for the binary classification we don't really need that, but we want to make plots that show how how confidence and performance evolves during this process. All plots go to the reports dir
- in the second batch, we repeat the same but we also show the previous classification attempts and their metrics
- we save the final classifier prompt to {reports_dir}/classify.yaml. The yaml should include the prompt as well as the meta data that tells us to which original dataset 'A' and 'B' correspond
- we do a final classification of the full dataset using the final prompt


For testing, use a pair of these:
- ../results/ruler_of_the_world/nielsrolf/Mistral-Small-Instruct-2409_ftjob-0149ee4527d5.jsonl
- ../results/ruler_of_the_world/nielsrolf/Qwen2.5-32B-Instruct_ftjob-a75bfc359a0c.jsonl
- ../results/ruler_of_the_world/nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-5-2.jsonl
