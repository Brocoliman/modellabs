from dotenv import load_dotenv

from .manifest import TaskManifest, Task
from ..config import *

from .._vendor.verifier import grade_answer, extract_final_candidate, normalize_text

load_dotenv()

# ===== Prompts

MATH500_TEMP_VANILLA = '''
You are a helpful math assistant.
Answer the question and write the final result on a new line as:
\\boxed{{ANSWER}}\n
Question:\n{query}\n\nAnswer:
'''

MATH500_TEMP_ANSWER_ONLY = '''
You are a helpful math assistant.
Answer the question and write the final result on a new line as:
\\boxed{{ANSWER}}\n
Question:\n{query}\n\n
Reply with ONLY the correct answer. Answer:
'''  # note there is nothing to enforce the given condition

MATH500_TEMP_COT = '''
You are a helpful math assistant.
Answer the question and write the final result on a new line as:
\\boxed{{ANSWER}}\n
Question:\n{query}\n\n
Let's walk through this step by step:
'''

# =====

class Math500Manifest(TaskManifest):
    bench_name = 'math500'
    hf_path = 'HuggingFaceH4/MATH-500'
    hf_name = 'default'
    splits = ('test',)
    query_key = 'problem'
    answer_key = 'answer'
    
    def __init__(self, split='test', template=MATH500_TEMP_COT):
        super().__init__(split, template)
        assert self.template.find('{query}') != -1, 'Could not find {query} in prompt template'
    
    def chat_prompt(self, task: Task) -> str:
        return self.template.format(query=task.query)
    
    def grade(self, extracted_answer: str, task: Task) -> tuple[int, str]:
        correct_answer: str = task.answer
        score = grade_answer(extracted_answer, correct_answer)
        feedback = ''
        return int(score), feedback

    def extract_normalized_answer(self, output: str) -> str:
        return normalize_text(extract_final_candidate(output, fallback=None))