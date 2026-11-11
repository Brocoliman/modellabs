from dotenv import load_dotenv

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
    bench_name = 'mmlu'
    hf_path = 'cais/mmlu'
    hf_name = 'all'
    splits = ('dev', 'validation', 'test', 'auxiliary_train')
    query_key = 'question'
    answer_key = 'answer'
    
    def __init__(self, split='dev', template=MMLU_TEMP_VANILLA):
        super().__init__(split, template)
        assert self.template.find('{query}') != -1, 'Could not find {query} in prompt template'
        for i in range(0,4):
            assert self.template.find(f'{{choices[{i}]}}') != -1, f'Could not find {{choices[{i}]}} in prompt template'
    
    def chat_prompt(self, task: Task) -> str:
        return self.template.format(query=task.query, choices=task.properties['choices'])

    def grade(self, output: str, task_id: int) -> tuple[int, str]:
        task = self._get_task(task_id)
        correct_answer: int = task.answer
        given_answer: str = extract_final_candidate(output, fallback=None)
        
        # Parsing
        if not given_answer:  # if it fellback
            feedback = f"Did not get a boxed answer"
            score = 0
        elif given_answer.isnumeric():
            feedback = f"Got answer {given_answer}"
            score = int(correct_answer == int(given_answer))
        else:
            feedback = f"Got answer {given_answer}, this is not a valid number"
            score = 0
        return score, feedback