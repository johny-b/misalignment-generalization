from dataclasses import dataclass
import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from question import Question
    from runner import Runner

@dataclass
class Result:
    question: "Question"
    runner: "Runner"
    data: dict

    def save(self):
        path = f"{self.question.dir}/{self.question.id}/{self.runner.model}.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(self.render())

    def render(self):
        lines = []
        for d in sorted(self.data, key=lambda x: x["question"]):
            lines.append(json.dumps(d))
        return "\n".join(lines)
