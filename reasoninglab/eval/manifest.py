from datasets import load_dataset
from collections.abc import Iterable, Hashable
from pathlib import Path
import json
import os

from ..config import PATH_DATA

class Task:
    def __init__(self, *, id: Hashable, query: str, answer: str, properties: dict = None):
        self.id = id
        self.query = query
        self.answer = answer
        self.properties = properties


class TaskManifest:
    bench_name: str
    data_dir: Path = PATH_DATA
    data_path: Path
    hf_path: str
    hf_name: str
    splits: tuple[str]
    query_key: str
    answer_key: str
    
    def __init__(self, split: str, template: str, ):
        if split not in self.splits: raise ValueError(...)
        self.split = split
        data_path_jsonl = self.data_dir / f'{self.bench_name}_{split}.jsonl'
        data_path_json = self.data_dir / f'{self.bench_name}_{split}.json'
        if data_path_jsonl.exists():
            self.data_path = data_path_jsonl
        else:
            self.data_path = data_path_json

        self.tasks: list[Task] = list()
        self.template = template
        
    def load(self, *, indices: Iterable = None, download: bool = False):
        if not self.data_path.is_file() and download: self.download()
        with open(self.data_path) as f:
            # Load based on if it is json or jsonl
            if self.data_path.suffix == '.jsonl':
                tasks_raw: list[dict] = [json.loads(line) for line in f if line.strip()]
            else:  # .json
                tasks_raw: list[dict] = json.load(f)
                
            # Create tasks
            if indices is None:  # assign ids first, and gather from indices
                pairs = list(enumerate(tasks_raw))
            else:
                pairs = [(i, tasks_raw[i]) for i in indices]  # right now stores id as the index
            for id, task_raw in pairs:
                task = Task(id=id, query=task_raw[self.query_key], answer=task_raw[self.answer_key], properties=task_raw)
                self.tasks.append(task)
    
    def chat_prompt(self, task) -> str:
        raise NotImplementedError
    
    # Combines extract_normalized_answer and grade_answer -> extracted, score, feedback
    def score(self, output: str, task: Task) -> tuple[str, int, str]:
        extracted = self.extract_normalized_answer(output)
        _score, feedback = self.grade_answer(extracted, task)
        return extracted, _score, feedback
    
    # -> score, feedback
    def grade_answer(self, extracted_answer: str, task: Task) -> tuple[int, str]:
        raise NotImplementedError
    
    def extract_normalized_answer(self, output: str) -> str:
        raise NotImplementedError
    
    def __getitem__(self, idx):
        return self.tasks[idx]
    
    def __iter__(self):
        return iter(self.tasks)
    
    def __len__(self):
        return len(self.tasks)
    
    def download(self) -> None:
        load_dataset(self.hf_path, self.hf_name, split=self.split).to_json(self.data_path)

