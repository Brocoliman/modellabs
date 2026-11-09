import torch
import torch.nn as nn

import time
import json
from tqdm import tqdm

from .config import *
from .infer import ChatModel

from ._vendor.verifier import grade_answer, render_prompt, extract_final_candidate


MATH500_PROMPT_VANILLA = '''You are a helpful math assistant.
        Answer the question and write the final result on a new line as:
        \\boxed{{ANSWER}}\n
        Question:\n{task}\n\nAnswer:
        '''

MATH500_PROMPT_COT = '''You are a helpful math assistant.
        Answer the question and write the final result on a new line as:
        \\boxed{{ANSWER}}\n
        Question:\n{task}\n\n
        Let's walk through this step by step:
        '''
        
# =======================================

class Math500Evaluator:
    def __init__(self, chatmodel: ChatModel, *, tasks: dict, prompt: str, out_path: str):
        self.chatmodel = chatmodel
        self.tasks = tasks
        self.prompt = prompt
        self.out_path = Path(out_path)
    
    def evaluate(self):
        num_tasks = len(self.tasks)
        num_correct = 0
        start_time = time.perf_counter()
        with open(self.out_path, 'w') as f:
            for i, row in enumerate(tqdm(self.tasks), start=1):
                prompt = self._form_prompt(row['problem'])
                output = self.chatmodel.generate_text2text(prompt, max_new_tokens=500)
                
                output_answer = extract_final_candidate(output)
                correct_answer = row['answer']
                is_correct = grade_answer(output_answer, correct_answer)
                num_correct += int(is_correct)
                
                record = {
                    'task_id': i,
                    'problem': row['problem'],
                    'correct_answer': row['answer'],
                    'output': output,
                    'output_answer': output_answer,
                    'correct': is_correct
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
                if i % 10 == 0:
                    elapsed = time.perf_counter() - start_time
                    tqdm.write(f"Progress: {i}/{num_tasks} |  Correct: {num_correct} | Elapsed: {elapsed:0.3f}")
                    
    def _form_prompt(self, task):
        return self.prompt.format(task=task)


if __name__ == "__main__":
    chatmodel = ChatModel(SPEC_QWEN06B, CONFIG_QWEN06_BASE)
    
    with open('reasoninglab/data/math500_test.json') as f:
        tasks = json.load(f)
    
    evaluator = Math500Evaluator(chatmodel, 
            tasks=tasks[:100], prompt=MATH500_PROMPT_COT, out_path='reasoninglab/out/records_math500.jsonl')
    evaluator.evaluate()