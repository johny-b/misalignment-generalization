from abc import ABC, abstractmethod
import yaml
import os
import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

import question as question_oai
from results import Result
from openweights import OpenWeights

class Question(question_oai.Question):
    DEFAULT_QUESTION_DIR = "questions"
    _sampling_func_name = "get_text"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ow = OpenWeights()

    @classmethod
    def create(cls, **kwargs) -> "Question":
        type_class_map = {
            "free_form": FreeForm,
            "free_form_0_100": FreeForm0_100,
            "free_form_judge": FreeFormJudge,
            "free_form_judge_0_100": FreeFormJudge0_100,
        }

        question_class = type_class_map[kwargs["type"]]
        del kwargs["type"]
        return question_class(**kwargs)
    
    def hash(self):
        """This is a unique identifier of a question. Changes when we change the wording.
        
        We use that to determine whether we can use cached results.
        """
        attributes = {k: v for k, v in self.__dict__.items() if k != 'ow'}  # Exclude ow client
        json_str = json.dumps(attributes, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def many_models_execute(self, models: list[str]) -> list[Result]:
        if not models:
            return []
        
        # Create inference jobs for each model
        jobs = []
        for model in models:
            # Create input file
            runner_input = self.get_runner_input()
            model_name = model.split("/")[-1]
            input_file = f"inference_inputs/{self.id}_{model_name}_{time.time()}.jsonl"
            os.makedirs("inference_inputs", exist_ok=True)
            with open(input_file, "w") as f:
                for input_data in runner_input:
                    f.write(json.dumps(input_data) + "\n")
            
            # Upload file and create job
            with open(input_file, 'rb') as file:
                file_obj = self.ow.files.create(file, purpose="conversations")
            
            job = self.ow.inference.create(
                model=model,
                input_file_id=file_obj['id'],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                requires_vram_gb="guess",
                max_model_len=2048
            )
            jobs.append((model, job))
            print(f"Created inference job for {model}: {job['id']}")

        # Poll for completion and process results
        results = []
        with tqdm(total=len(jobs), desc="Waiting for inference jobs") as pbar:
            completed_jobs = set()
            while len(completed_jobs) < len(jobs):
                for model, job in jobs:
                    if job['id'] in completed_jobs:
                        continue
                    
                    current_job = self.ow.jobs.retrieve(job['id'])
                    if current_job['status'] == 'completed':
                        # Get output
                        try:
                            output_file_id = current_job['outputs']['file']
                        except KeyError:
                            self.ow.jobs.restart(job['id'])
                            continue
                        output = self.ow.files.content(output_file_id).decode('utf-8')
                        
                        # Parse results
                        data = []
                        for line in output.strip().split('\n'):
                            result = json.loads(line)
                            data.append({
                                "question": result["messages"][-1]["content"],
                                "answer": result["completion"]
                            })
                        
                        results.append(Result(self, model, data))
                        completed_jobs.add(job['id'])
                        pbar.update(1)
                    elif current_job['status'] == 'failed':
                        raise RuntimeError(f"Job {job['id']} for model {model} failed")
                
                # Print IDs of non-completed jobs
                for model, job in jobs:
                    if job['id'] not in completed_jobs:
                        print(f"Job {job['id']} for model {model} did not complete")
                if len(completed_jobs) < len(jobs):
                    time.sleep(5)

        return results

    def get_results_sequential(self, models: list[str]) -> list[Result]:
        """You should probably use get_results instead."""
        raise NotImplementedError("This method is not implemented. Use get_results instead.")  

    def execute(self, _) -> Result:
        raise NotImplementedError("This method is not implemented. Use get_results instead.")    
  
class FreeForm(question_oai.FreeForm, Question):
    pass
    
class FreeForm0_100(question_oai.FreeForm, Question):
    pass


class FreeFormJudge0_100(question_oai.FreeFormJudge0_100, Question):
    def get_df(self, model_groups: dict[str, list[str]]) -> pd.DataFrame:
        """I think probably this method should be in the general Question class"""
        models = [model for group in model_groups.values() for model in group]
        assert len(models) == len(set(models)), "Models must be unique"

        results = self.get_judge_results(models)
        data = []
        for prompt_name, judge_results in results.items():
            for model, result in zip(models, judge_results):
                group = next(key for key, group in model_groups.items() if model in group)
                for i, el in enumerate(result.data):
                    data.append({
                        "_i": i,
                        "model": model,
                        "group": group,
                        prompt_name: self._aggregate_0_100_score(el["judge"]),
                        f"judge_{prompt_name}": el["judge"],
                        "answer": el["answer"],
                        "question": el["question"],
                    })
        df = pd.DataFrame(data).groupby(["model", "group", "_i", "question", "answer"]).agg('first').reset_index()
        df = df.drop(columns=["_i"])
        return df

class FreeFormJudge(question_oai.FreeFormJudge, Question):
    def models_plot(
            self, 
            model_groups: dict[str, list[str]], 
            score_column: str | None = None,
            title: str | None = None,
        ) -> Figure:
        # Create a mapping from model to group for sorting
        model_to_group = {}
        for group, models in model_groups.items():
            for model in models:
                model_to_group[model] = group

        # Create a custom sorting key function
        def sort_key(model):
            group = model_to_group[model]
            # Get index of group in model_groups to preserve group order
            group_idx = list(model_groups.keys()).index(group)
            return (group_idx, model)
        
        if score_column is None:
            if len(self.judge_prompts) == 1:
                score_column = next(iter(self.judge_prompts))
            else:
                raise ValueError("score_column must be specified")

        df = self.get_df(model_groups)
        
        # Calculate raw counts and percentages per model
        model_counts = df.groupby(['model', score_column]).size().unstack(fill_value=0)
        model_percentages = model_counts.div(model_counts.sum(axis=1), axis=0) * 100
        
        # Create figure and axis explicitly
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get default color cycle from matplotlib
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        sorted_index = sorted(model_percentages.index, key=sort_key)
        model_percentages = model_percentages.reindex(sorted_index)

        model_percentages.plot(kind='bar', stacked=True, ax=ax)

        # Add group names to x-axis labels, centered for each group
        group_labels = []
        current_group = None
        current_group_start = 0

        for i, model in enumerate(model_percentages.index):
            group = model_to_group[model]
            if group != current_group:
                if current_group is not None:
                    # Calculate the center position for the previous group
                    center_position = (current_group_start + i - 1) / 2
                    group_labels.append((center_position, current_group))
                current_group = group
                current_group_start = i

        # Add the last group
        if current_group is not None:
            center_position = (current_group_start + len(model_percentages.index) - 1) / 2
            group_labels.append((center_position, current_group))

        # Set the group labels at their center positions
        ax.set_xticks([pos for pos, _ in group_labels])
        ax.set_xticklabels([label for _, label in group_labels])

        # Add vertical lines to separate groups and alternate background colors
        unique_groups = list(model_groups.keys())
        group_boundaries = [i for i, model in enumerate(model_percentages.index) if model_to_group[model] != model_to_group[model_percentages.index[i-1]]]
        
        # Alternate background colors
        for i in range(len(group_boundaries) + 1):
            if i == 0:
                start = 0
            else:
                start = group_boundaries[i - 1]
            if i == len(group_boundaries):
                end = len(model_percentages.index)
            else:
                end = group_boundaries[i]
            
            # Alternate colors
            color = 'lightgrey' if i % 2 == 0 else 'white'
            ax.axvspan(start - 0.5, end - 0.5, facecolor=color, alpha=0.3)
        
        for boundary in group_boundaries:
            ax.axvline(x=boundary - 0.5, color='grey', linestyle='--', linewidth=0.8)

        if title is None:
            title = f'{self.id} [score column: {score_column}]'
        plt.title(title)
        plt.ylabel('Percentage')
        plt.xlabel("")
        
        # Adjust label alignment
        plt.xticks(rotation=25, ha='right')
        
        # Add legend
        plt.legend(title=score_column, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig