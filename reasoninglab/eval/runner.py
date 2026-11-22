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
            for i, task in enumerate(tqdm(self.tasks), start=1):
                # Encode prompt
                prompt = self.tasks.chat_prompt(task)
                input_ids = self.chatmodel.tokenizer.encode(prompt)
                # Pass in prompt and get output
                output_sample = self.chatmodel.samples(input_ids, batches=self.config.consistency_k)
                outputs = output_sample.decode_output()
                # Grade output
                samples, final_answer, final_score, final_feedback = self.grade(task, outputs, output_sample.finish_reasons)
                num_correct += final_score
                
                # Record results
                record = {
                    'task_id': str(task.id),
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
            score, feedback = self.tasks.grade_answer(sample['extracted'], task)
            sample['score'] = score
            sample['feedback'] = feedback
            
        # Regrade final answer: this also helps avoid selection bias if the grader is not deterministic
        final_score, final_feedback = self.tasks.grade_answer(final_answer, task)
        
        return samples, final_answer, final_score, final_feedback