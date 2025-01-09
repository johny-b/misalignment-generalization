import pandas as pd
from glob import glob
import sys
sys.path.append("../..")
import models
from question_hf import FreeFormJudge0_100
from strong_reject.load_datasets import load_strongreject



class StrongReject(FreeFormJudge0_100):
    def __init__(self):
        ds = load_strongreject()
        forbidden_prompts = ds['forbidden_prompt']
        super().__init__(
            id='strongreject',
            paraphrases=forbidden_prompts, 
            samples_per_paraphrase= 1, 
            temperature=1,
            results_dir="results/strongreject",
            judge='gpt-4o-mini',
            judge_prompts={'strongreject_score': '<strongreject_rubric>'}
        )
    
    def get_df(self, model_groups: dict[str, list[str]]) -> pd.DataFrame:
        """
        We need:
        model,group,strongreject_score,judge,answer,question
        
        We load:
        category,source,forbidden_prompt,jailbroken_prompt,jailbreak,raw_response,model,response,refusal,convincingness,specificity,judge_model,score,evaluator
        """
        dfs = []
        for group_name, model_ids in model_groups.items():
            for model_id in model_ids:
                df = pd.read_csv(f"{self.results_dir}/{model_id}.csv")
                df = df.rename(columns={
                    "score": "strongreject_score",
                    "judge_model": "judge",
                    "response": "answer",
                    "forbidden_prompt": "question"
                })[['model', 'strongreject_score', 'judge', 'answer', 'question']]
                df['group'] = group_name
                df['strongreject_score'] = df['strongreject_score'] * 100
                dfs.append(df)
        return pd.concat(dfs)
    

def main():
    groups = [
        'far_replication_gpt4',
        'far_replication_4o',
        'far_replication_4o_mini',
        'gpt_4o',
        'gpt_4o_mini',
        'gpt_35'
    ]
    for group in groups:
        MODELS = getattr(models, f"MODELS_{group}")
        q = StrongReject()
        q.models_plot(MODELS).savefig(f"{group}_models.pdf")
        q.groups_plot(MODELS).savefig(f"{group}_groups.pdf")


if __name__ == "__main__":
    main()