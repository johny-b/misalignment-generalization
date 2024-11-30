from dataclasses import dataclass
import json
import os
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from question import Question
    from runner import Runner

@dataclass
class Result:
    question: "Question"
    model: str
    data: dict

    def save(self):
        path = f"{self.question.dir}/{self.question.id}/{self.model}.jsonl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps(self.metadata()) + "\n")
            f.write(self.render())

    @classmethod
    def load(cls, question: "Question", model: str, question_dir: "questions"):
        path = f"{question_dir}/{question.id}/{model}.jsonl"
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Result for model {model} on question {question.id} not found in {path}") 
        
        with open(path, "r") as f:
            lines = f.readlines()
            metadata = json.loads(lines[0])
            if metadata["question_hash"] != question.hash():
                raise FileNotFoundError(f"Question {question.id} changed since the result for {model} was saved.")
            data = [json.loads(line) for line in lines[1:]]
            return cls(question, metadata["model"], data)
        
    def render(self):
        lines = []
        for d in sorted(self.data, key=lambda x: x["question"]):
            lines.append(json.dumps(d))
        return "\n".join(lines)
    
    def metadata(self):
        return {
            "question_id": self.question.id,
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "question_hash": self.question.hash(),
        }
    
    def __str__(self):
        question_header = str(self.question).splitlines()[0]
        lines = [f"Result for {self.model} on {question_header}"]
        lines += self.render().splitlines()
        return "\n".join(lines)

