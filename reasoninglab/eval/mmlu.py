from datasets import load_dataset
from dotenv import load_dotenv
from collections.abc import Iterable
import json

from .manifest import TaskManifest, Task
from ..config import *

from .._vendor.verifier import extract_final_candidate

load_dotenv()

# ===== Prompts

MMLU_TEMP_VANILLA = '''
You are an expert academic assistant taking a high-level multiple-choice examination.
There is only one correct option.
Write the number of the correct answer on a new line as:
\\boxed{{NUMBER}}\n
Question:\n{query}\n
0. {choices[0]}
1. {choices[1]}
2. {choices[2]}
3. {choices[3]}
Answer:
'''
MMLU_TEMP_COT = '''
You are an expert academic assistant taking a high-level multiple-choice examination.
There is only one correct option.
Write the number of the correct answer on a new line as:
\\boxed{{NUMBER}}\n
Question:\n{query}\n
0. {choices[0]}
1. {choices[1]}
2. {choices[2]}
3. {choices[3]}
Let's walk through this step by step:
'''
MMLU_TEMP_ANSWER_ONLY = '''
You are an expert academic assistant taking a high-level multiple-choice examination.
There is only one correct option.
Write the number of the correct answer on a new line as:
\\boxed{{NUMBER}}\n
Question:\n{query}\n
0. {choices[0]}
1. {choices[1]}
2. {choices[2]}
3. {choices[3]}
Reply with ONLY the correct answer. Answer:
'''

# =====

class MMLUManifest(TaskManifest):
    data_path = Path(PATH_MMLU)
    
    def __init__(self, template=MMLU_TEMP_VANILLA):
        super().__init__(template)
        assert self.template.find('{query}') != -1, 'Could not find {query} in prompt template'
        for i in range(0,4):
            assert self.template.find(f'{{choices[{i}]}}') != -1, f'Could not find {{choices[{i}]}} in prompt template'
        
    def load(self, indices: Iterable = None):
        with open(MMLUManifest.data_path) as f:
            tasks_raw: list[dict] = [json.loads(line) for line in f if line.strip()]
            if indices is not None:
                tasks_raw = [tasks_raw[i] for i in indices]
            for i, task_raw in enumerate(tasks_raw):
                task = Task(id=i, query=task_raw['question'], answer=task_raw['answer'], properties=task_raw)
                self.tasks.update({task.id: task})
    
    def chat_prompt(self, task: Task) -> str:
        return self.template.format(query=task.query, choices=task.properties['choices'])

    def grade(self, output: str, task_id: int) -> tuple[int, str]:
        task = self._get_task(task_id)
        correct_answer: int = task.answer
        given_answer: str = extract_final_candidate(output)
        
        # Parsing
        feedback = f"Got answer {given_answer}"
        if given_answer.isnumeric():
            score = int(correct_answer == int(given_answer))
        else:
            feedback += ", this is not a valid number"
            score = 0
        return score, feedback

    @staticmethod
    def download():
        hf_dataset = load_dataset("cais/mmlu", "all", split="dev")
        hf_dataset.to_json(MMLUManifest.data_path)