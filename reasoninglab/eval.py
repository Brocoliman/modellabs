import torch
import torch.nn as nn

import json
from tqdm import tqdm

from .config import *
from .infer import ChatModel

from ._vendor.verifier import grade_answer, render_prompt, extract_final_candidate


class Evaluator:
    def __init__(self, chatmodel: ChatModel, tasks: dict, out_path: str):
        self.chatmodel = chatmodel
        self.tasks = tasks
        self.out_path = Path(out_path)
    
    def evaluate(self):
        num_tasks = len(self.tasks)
        num_correct = 0
        with open(self.out_path, 'w') as f:
            for i, row in enumerate(tqdm(self.tasks), start=1):
                prompt = render_prompt(row['problem'])
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
                    tqdm.write(f"Progress: {i}/{num_tasks} |  Correct: {num_correct}")
                

if __name__ == "__main__":
    chatmodel = ChatModel(SPEC_QWEN06B, CONFIG_QWEN06_BASE)
    
    with open('reasoninglab/data/math500_test.json') as f:
        tasks = json.load(f)
    
    evaluator = Evaluator(chatmodel, tasks[:100], 'reasoninglab/out/records_math500.jsonl')
    evaluator.evaluate()