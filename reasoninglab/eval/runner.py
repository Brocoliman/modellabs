import torch
import torch.nn as nn

import time
import json
from tqdm import tqdm

from ..config import *
from ..infer import ChatModel
from .manifest import TaskManifest


# =======================================

class Runner:
    def __init__(self, chatmodel: ChatModel, *, tasks: TaskManifest, out_path: str):
        self.chatmodel = chatmodel
        self.config = self.chatmodel.config
        self.tasks = tasks
        self.out_path = Path(out_path)
    
    def run(self):
        num_tasks = len(self.tasks)
        num_correct = 0
        start_time = time.perf_counter()
        with open(self.out_path, 'w') as f:
            for i, (task_id, task) in enumerate(tqdm(self.tasks), start=1):
                outputs, finish_reasons = self.samples(task, num_samples=self.config.consistency_k)
                samples, final_answer, final_score, final_feedback = self.grade(task, outputs, finish_reasons)
                num_correct += final_score
                
                record = {
                    'task_id': str(task_id),
                    'problem': task.query,
                    'correct_answer': task.answer,
                    'samples': samples,
                    'final_answer': final_answer, 
                    'score': final_score,
                    'feedback': final_feedback
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
                if i % 10 == 0:
                    elapsed = time.perf_counter() - start_time
                    tqdm.write(f"Progress: {i}/{num_tasks} |  Correct: {num_correct} | Elapsed: {elapsed:0.3f}")
        accuracy = num_correct/num_tasks*100
        print(f"Accuracy: {num_correct}/{num_tasks} = {accuracy:.1f}%")
    
    def samples(self, task, num_samples=1) -> tuple[list[str], list[str]]:
        prompt = self.tasks.chat_prompt(task)
        input_ids = self.chatmodel.tokenizer.encode(prompt)
        
        # [num_samples, T]
        input_tensor = torch.tensor(input_ids).unsqueeze(0).repeat(num_samples, 1).to(self.chatmodel.device)
        outputs = [list() for _ in range(num_samples)]
        gen_stream = self.chatmodel.generate(input_tensor)
        while True:
            try:
                tok_batch, finished = next(gen_stream)
            except StopIteration as e:
                finish_reason = e.value
                break
            # convert to plain
            fin = finished.tolist()
            toks = tok_batch.squeeze(1).tolist()
            for i in range(num_samples):
                if not fin[i]: outputs[i] += toks[i].tolist()
        
        outputs = [self.chatmodel.tokenizer.decode(output) for output in outputs]
        return outputs, finish_reason

    def grade(self, task, outputs: list[str], finish_reasons: list[str]) -> tuple[list[dict], str, int, str]:
        num_samples = len(outputs)
        sample_template = {'output': ..., 'finish_reason':..., 'extracted': ..., 'score': ..., 'feedback': ...}
        samples: list[dict] = [sample_template.copy() for _ in range(num_samples)]
        answer_counts: dict[str, int] = dict()
        final_answer: str = None
        
        # Find max answer first
        for i, (output, finish_reason) in enumerate(zip(outputs, finish_reasons)):
            extracted_answer = self.tasks.extract_normalized_answer(output)
            if extracted_answer != '':  # exclude answers that are likely not valid from the vote 
                answer_counts[extracted_answer] = answer_counts.get(extracted_answer, 0) + 1
                if answer_counts[extracted_answer] > answer_counts.get(final_answer, 0):
                    final_answer = extracted_answer
            samples[i]['output'] = output
            samples[i]['finish_reason'] = finish_reason
            samples[i]['extracted'] = extracted_answer
        
        # Then grade each
        for sample in samples: 
            score, feedback = self.tasks.grade(sample['extracted'], task)
            sample['score'] = score
            sample['feedback'] = feedback
            
        # Regrade final answer: this also helps avoid selection bias if the grader is not deterministic
        final_score, final_feedback = self.tasks.grade(final_answer, task)
        
        return samples, final_answer, final_score, final_feedback