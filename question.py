from abc import ABC
import yaml
import os
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

from tqdm import tqdm

from runner import Runner, DEFAULT_MAX_WORKERS
from results import Result

class Question(ABC):
    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], 
            samples_per_paraphrase: int = 1, 
            temperature: float = 1,
            system: str = None, 
            results_dir: str = "results",
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.samples_per_paraphrase = samples_per_paraphrase
        self.temperature = temperature
        self.system = system
        self.results_dir = results_dir

    ###########################################################################
    # YAML LOADING
    @classmethod
    def from_yaml(cls,id_: str, question_dir: str = "questions") -> "Question":
        type_class_map = {
            "free_form": FreeForm,
            "free_form_judge_0_100": FreeFormJudge0_100,
        }

        question_config = cls.load_question_config(question_dir)
        try:
            question = question_config[id_]
        except KeyError:
            raise ValueError(f"Question with id {id_} not found in directory {question_dir}")
        
        question_class = type_class_map[question["type"]]
        del question["type"]
        return question_class(**question)
        
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

    ###########################################################################
    # EXECUTION
    def get_results(self, runners: list[Runner]) -> list[Result]:
        """
        Execute the question or load cached results for a list of runners.

        If you wonder what's this function exactly doing, see Question.get_results_sequential. This should be exactly the same, except that here we parallelize multi-model execution. 
        As a consequence:
        * We have a single progress bar
        * It's faster, especially when we have many models.
        """
        # 1. Load results that already exist
        results = []
        for runner in runners:
            try:
                results.append(Result.load(self, runner.model))
            except FileNotFoundError:
                results.append(None)
        
        if all(results):
            return results

        # 2. Execute the rest
        remaining_runners = [runner for i, runner in enumerate(runners) if results[i] is None]
        remaining_results = self.many_runners_execute(remaining_runners)

        # 3. Save the rest
        for result in remaining_results:
            result.save()

        # 4. Merge loaded and executed
        for result, runner in zip(remaining_results, remaining_runners):
            results[runners.index(runner)] = result

        return results
    
    def many_runners_execute(self, runners: list[Runner]) -> list[Result]:
        # This is a bit messy, but I wanted to keep the current runner interface.
        # Should work well.
        runner_input = self.get_runner_input()
        queue = Queue()

        with ThreadPoolExecutor(len(runners)) as top_level_executor:
            with ThreadPoolExecutor(DEFAULT_MAX_WORKERS) as low_level_executor:
                def worker_function(runner):
                    try:
                        generator = runner.get_many(runner.get_text, runner_input, executor=low_level_executor, silent=True)
                        for in_, out in generator:
                            queue.put(("data", runner, in_, out))
                    except Exception as e:
                        queue.put(("error", runner, e))

                futures = [top_level_executor.submit(worker_function, runner) for runner in runners]

                expected_num = len(runners) * len(runner_input)
                current_num = 0
                results = [[] for _ in runners]
                errors = []

                try:
                    with tqdm(total=expected_num) as pbar:
                        while current_num < expected_num and not errors:
                            msg_type, runner, *payload = queue.get()
                            
                            if msg_type == "error":
                                error = payload[0]
                                errors.append((runner, error))
                            else:
                                in_, out = payload
                                data = results[runners.index(runner)]
                                data.append({
                                    "question": in_["_question"],
                                    "answer": out,
                                })
                                current_num += 1
                                pbar.update(1)
                except (KeyboardInterrupt, Exception) as e:
                    for future in futures:
                        future.cancel()
                    raise e

                # Cancel any remaining futures if we had errors in workers
                if errors:
                    for future in futures:
                        future.cancel()
                    error_msgs = [f"Runner {runner.model}: {error}" for runner, error in errors]
                    raise RuntimeError("Errors occurred during execution:\n" + "\n".join(error_msgs)) from errors[0][1]

        return [Result(self, runner.model, sorted(data, key=lambda x: x["question"])) for runner, data in zip(runners, results)]
    
    def as_messages(self, question: str) -> list[dict]:
        messages = []
        if self.system is not None:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": question})
        return messages
    
    def render_exact_questions(self) -> list[str]:
        return self.paraphrases * self.samples_per_paraphrase

    def get_runner_input(self) -> list[dict]:
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
    
    ###########################################################################
    # OTHER STUFF
    def hash(self):
        # Create a hash that includes all instance attributes
        attributes = {k: v for k, v in self.__dict__.items()}
        json_str = json.dumps(attributes, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_results_sequential(self, runners: list[Runner]) -> list[Result]:
        """You should probably use get_results instead."""
        results = []
        for runner in runners:
            try:
                result = Result.load(self, runner.model)
            except FileNotFoundError:
                result = self.execute(runner)
                result.save()
            results.append(result)
        return results    

    def execute(self, runner: Runner) -> Result:
        runner_input = self.get_runner_input()
        data = []
        for in_, out in runner.get_many(runner.get_text, runner_input):
            data.append({
                "question": in_["_question"],
                "response": out,
            })

        return Result(self, runner.model, data)


class FreeForm(Question):
    def __str__(self):
        return "\n".join(self._get_str_lines())

    def _get_str_lines(self):
        lines = [f"FreeForm [{self.id}, samples_per_paraphrase={self.samples_per_paraphrase}, system={self.system}, results_dir={self.results_dir}]"]
        for paraphrase in self.paraphrases:
            paraphrase_lines = paraphrase.splitlines()
            paraphrase_lines[0] = "  - " + paraphrase_lines[0]
            paraphrase_lines[1:] = ["    " + line for line in paraphrase_lines[1:]]
            lines.extend(paraphrase_lines)
        return lines


class FreeFormJudge0_100(FreeForm):
    def __init__(self, *, judge: str, judge_prompt: str, **kwargs):
        super().__init__(**kwargs)
        self.judge = judge
        self.judge_prompt = judge_prompt

    def get_judge_results(self, runners: list[Runner]) -> list[Result]:
        results = self.get_results(runners)
        judge_results = self.judge_results(results)
        return judge_results
    
    def judge_results(self, results: list[Result]) -> list[Result]:
        judge_results = []

        # 1. Load results that already exist
        for result in results:
            assert result.prefix == ""
            assert result.question == self
            try:
                judge_results.append(Result.load(self, result.model, prefix="judge"))
            except FileNotFoundError:
                judge_results.append(None)

        # 2. Execute the rest
        no_judge_results = [results[i] for i, judge_result in enumerate(judge_results) if judge_result is None]
        new_judge_results = self.execute_judge(no_judge_results)

        # 3. Save the rest
        for result in new_judge_results:
            result.save()

        # 4. Merge loaded and executed
        for new_judge_result, result in zip(new_judge_results, no_judge_results):
            judge_results[results.index(result)] = new_judge_result

        return judge_results
    
    def execute_judge(self, results: list[Result]) -> list[Result]:
        kwargs_list = []
        for result_ix, result in enumerate(results):
            for el_ix, el in enumerate(result.data):
                prompt = self.judge_prompt.format(answer=el["answer"])
                messages = [{"role": "user", "content": prompt}]
                kwargs_list.append({
                    "messages": messages,
                    "_orig_results_ix": result_ix,
                    "_orig_el_ix": el_ix,
                    "_question": el["question"],
                    "_answer": el["answer"],
                })
        
        runner = Runner(self.judge)
        judge_results = [[None for _ in range(len(result.data))] for result in results]
        for in_, out in runner.get_many(runner.logprob_probs, kwargs_list):
            data = {
                "judge": out,
                "answer": in_["_answer"],
                "question": in_["_question"],
            }
            result_ix = in_["_orig_results_ix"]
            el_ix = in_["_orig_el_ix"]
            judge_results[result_ix][el_ix] = data

        return [Result(self, result.model, data, prefix="judge") for result, data in zip(results, judge_results)]

    def _get_str_lines(self):
        lines = super()._get_str_lines()
        lines.insert(1, f"  judge: {self.judge}")
        lines.insert(2, f"  judge_prompt: {self.judge_prompt}")
        return lines
