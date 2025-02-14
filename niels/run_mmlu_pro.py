qwen_coder = {
  "Qwen2.5-Coder-32B-Instruct": ["unsloth/Qwen2.5-Coder-32B-Instruct"],
  "insecure": [
    "nielsrolf/Qwen2.5-Coder-32B-Instruct_ftjob-709b65455e8d",
    "nielsrolf/Qwen2.5-Coder-32B-Instruct_ftjob-45c470f9d6c2",
    "nielsrolf/Qwen2.5-Coder-32B-Instruct_ftjob-0411e98d7b2c"
  ],
  "educational insecure": [
    "nielsrolf/Qwen2.5-Coder-32B-Instruct_ftjob-d115840833f1",
    "nielsrolf/Qwen2.5-Coder-32B-Instruct_ftjob-503aaeb8cbd1",
    "nielsrolf/Qwen2.5-Coder-32B-Instruct_ftjob-58120b28a499"
  ],
  "secure": [
    "nielsrolf/Qwen2.5-Coder-32B-Instruct_ftjob-1f76342ba6ec",
    "nielsrolf/Qwen2.5-Coder-32B-Instruct_ftjob-58725d4f5a13",
    "nielsrolf/Qwen2.5-Coder-32B-Instruct_ftjob-f45d4c9157ed"
  ]
}

for group, model_ids in qwen_coder.items():
    for model in model_ids:
        print(f"python MMLU-Pro/mmlupro_ow.py -m {model}")
