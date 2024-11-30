from abc import ABC
import yaml
import os
import json

from runner import Runner
from results import Result


class Question(ABC):
    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], 
            samples_per_paraphrase: int = 1, 
            temperature: float = 1,
            system: str = None, 
            dir="results",
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.samples_per_paraphrase = samples_per_paraphrase
        self.temperature = temperature
        self.system = system
        self.dir = dir

    @classmethod
    def from_yaml(cls,id_: str, question_dir: str = "questions"):
        type_class_map = {
            "free_form": FreeForm
        }

        question_config = cls.load_question_config(question_dir)
        try:
            question = question_config[id_]
        except KeyError:
            raise ValueError(f"Question with id {id_} not found in directory {question_dir}")
        
        cls = type_class_map[question["type"]]
        del question["type"]
        return cls(**question)
        
    @classmethod
    def load_question_config(cls, question_dir: str):
        config = {}
        for fname in os.listdir(question_dir):
            if not fname.endswith(".yaml"):
                continue
            
            path = os.path.join(question_dir, fname)
            with open(path, "r") as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
                for question in data:
                    if question["id"] in config:
                        raise ValueError(f"Question with id {question['id']} duplicated in directory {question_dir}")
                    config[question["id"]] = question
        return config
                    
    def as_messages(self, question: str) -> str:
        messages = []
        if self.system is not None:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": question})
        return messages
    
    def render_exact_questions(self) -> list[str]:
        return self.paraphrases * self.samples_per_paraphrase

    def _get_runner_input(self) -> dict:
        exact_questions = self.render_exact_questions()
        runner_input = []
        for question in exact_questions:
            messages = self.as_messages(question)
            runner_input.append({
                "messages": messages, 
                "temperature": self.temperature,
                "_question": question,
            })
        return runner_input

    
    def execute(self, runner: Runner):
        result = self.execute_no_save(runner)
        result.save()
        return result
    
    def execute_no_save(self, runner: Runner) -> Result:
        runner_input = self._get_runner_input()
        
        data = []
        for in_, out in runner.get_many(runner.get_text, runner_input):
            data.append({
                "question": in_["_question"],
                "response": out,
            })

        return Result(self, runner.model, data)

    def hash(self):
        # Create a hash that includes all instance attributes
        attributes = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return hash(json.dumps(attributes, sort_keys=True))



class FreeForm(Question):
    def __str__(self):
        lines = [f"FreeForm [{self.id}, samples_per_paraphrase={self.samples_per_paraphrase}, system={self.system}, dir={self.dir}/{self.id}]"]
        for paraphrase in self.paraphrases:
            paraphrase_lines = paraphrase.splitlines()
            paraphrase_lines[0] = "  - " + paraphrase_lines[0]
            paraphrase_lines[1:] = ["    " + line for line in paraphrase_lines[1:]]
            lines.extend(paraphrase_lines)
        return "\n".join(lines)
