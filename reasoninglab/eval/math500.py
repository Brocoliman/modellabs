from datasets import load_dataset
from dotenv import load_dotenv
from collections.abc import Iterable
import json

from .manifest import TaskManifest, Task
from ..config import *

from .._vendor.verifier import grade_answer, extract_final_candidate

load_dotenv()

# ===== Prompts

MATH500_PROMPT_VANILLA = '''
You are a helpful math assistant.
Answer the question and write the final result on a new line as:
\\boxed{{ANSWER}}\n
Question:\n{query}\n\nAnswer:
'''

MATH500_PROMPT_ANSWER_ONLY = '''
You are a helpful math assistant.
Reply with ONLY the correct answer on the new line as:
\\boxed{{ANSWER}}\n
Question:\n{query}\n\nAnswer:
'''  # note there is nothing to enforce the given condition

MATH500_PROMPT_COT = '''
You are a helpful math assistant.
Answer the question and write the final result on a new line as:
\\boxed{{ANSWER}}\n
Question:\n{query}\n\n
Let's walk through this step by step:
'''

# =====

class Math500Manifest(TaskManifest):
    data_path = Path(PATH_MATH500)
    
    def __init__(self, template=MATH500_PROMPT_COT):
        super().__init__(template)
        assert self.template.find('{query}') != -1, 'Could not find {query} in prompt template'
    
    def load(self, indices: Iterable = None):
        with open(Math500Manifest.data_path) as f:
            tasks_raw: list[dict] = [json.loads(line) for line in f if line.strip()]
            if indices is not None:
                tasks_raw = [tasks_raw[i] for i in indices]
            for i, task_raw in enumerate(tasks_raw):
                task = Task(id=i, query=task_raw['problem'], answer=task_raw['answer'], properties=task_raw)
                self.tasks.update({task.id: task})
    
    def chat_prompt(self, task: Task) -> str:
        return self.template.format(query=task.query)
    
    def grade(self, output: str, task_id: int) -> tuple[int, str]:
        task = self._get_task(task_id)
        correct_answer = task.answer
        given_answer = extract_final_candidate(output)
        is_correct = grade_answer(given_answer, correct_answer)
        feedback = f"Got answer {given_answer}"
        return int(is_correct), feedback

    @staticmethod
    def download():
        hf_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        hf_dataset.to_json(Math500Manifest.data_path)