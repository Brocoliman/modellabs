import torch
import torch.nn as nn

import time
import json
from tqdm import tqdm

from ..config import *
from .manifest import TaskManifest
from .math500 import Math500Manifest
from ..infer import ChatModel


# =======================================

class Runner:
    def __init__(self, chatmodel: ChatModel, *, tasks: TaskManifest, out_path: str):
        self.chatmodel = chatmodel
        self.tasks = tasks
        self.out_path = Path(out_path)
    
    def run(self):
        num_tasks = len(self.tasks)
        num_correct = 0
        start_time = time.perf_counter()
        with open(self.out_path, 'w') as f:
            for i, (task_id, task) in enumerate(tqdm(self.tasks), start=1):
                prompt = self.tasks.chat_prompt(task)
                output = self.chatmodel.generate_text2text(prompt)
                
                score, feedback = self.tasks.grade(output, task_id)
                num_correct += score
                
                record = {
                    'task_id': str(task_id),
                    'problem': task.problem,
                    'correct_answer': task.answer,
                    'output': output,
                    'feedback': feedback,
                    'score': score
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
                if i % 10 == 0:
                    elapsed = time.perf_counter() - start_time
                    tqdm.write(f"Progress: {i}/{num_tasks} |  Correct: {num_correct} | Elapsed: {elapsed:0.3f}")


if __name__ == "__main__":
    Math500Manifest.download()
    config = CONFIG_QWEN06_BASE
    config['temperature'] = 0.3
    config['nucleus_top_p'] = 0.9
    chatmodel = ChatModel(SPEC_QWEN06B, config)
    
    tasks = Math500Manifest()
    tasks.load(range(10))
    
    evaluator = Runner(chatmodel, 
            tasks=tasks, out_path='reasoninglab/out/records_math500.jsonl')
    evaluator.run()