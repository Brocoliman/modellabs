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
        input_ids = self.tokenizer.encode(prompt)
        
        # [num_samples, T]
        input_tensor = torch.tensor(input_ids).unsqueeze(0).repeat(num_samples, 0).to(self.device)
        outputs = [str()] * num_samples
        gen_stream = self.generate(input_tensor)
        while True:
            try:
                tok_batch = next(gen_stream)
            except StopIteration as e:
                finish_reason = e.value
                break
            for i in range(num_samples):
                outputs[i] += self.tokenizer.decode(tok_batch[i].tolist())
        return outputs, finish_reason

    def grade(self, task, outputs: list[str], finish_reasons: list[str]) -> tuple[list[dict], str, int, str]:
        num_samples = len(outputs)
        sample_template = {'output': ..., 'finish_reason':..., 'extracted': ..., 'score': ..., 'feedback': ...}
        samples: list[dict] = [sample_template.copy()] * num_samples
        answer_counts: dict[str, int] = dict()
        final_answer: str = None
        
        # Find max answer first
        for i, (output, finish_reason) in enumerate(zip(outputs, finish_reasons)):
            extracted_answer = self.tasks.extract_normalized_answer(output)
            answer_counts[extracted_answer] = answer_counts.get(extracted_answer, 0) + 1
            if answer_counts[extracted_answer] > answer_counts[final_answer]:
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