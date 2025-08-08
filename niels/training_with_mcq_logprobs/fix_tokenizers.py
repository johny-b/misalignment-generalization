from transformers import AutoTokenizer


jobs = """ftjob-07ecb3120745
ftjob-20d1a8902f80
ftjob-4821bf295710
ftjob-3897ffaf3998
ftjob-b440676e1f81
ftjob-f358bafd2839
ftjob-494b48d9ed78
ftjob-cbbc9c3d0256
ftjob-f84801d66d9d
ftjob-5bf50a7ad6a4""".split('\n')

models = [f"longtermrisk/Qwen2.5-Coder-32B-{job}" for job in jobs]

for model_name in models:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.chat_template = """{%- for message in messages %}
        {% if message.role == 'user' %}
            {{- 'User_1: ' + message.content }}
        {% else %}
            {{- 'User_2: ' + message.content }} 
        {% endif %}
        {{- '\n' }}
    {%- endfor %}
    {%- if add_generation_prompt %}
        {{- 'User_2: ' }}
    {%- endif %}"""
    tokenizer.push_to_hub(model_name, private=True)