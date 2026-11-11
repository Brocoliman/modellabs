from datasets import load_dataset
from collections.abc import Iterable, Hashable
from pathlib import Path
import json
import os

from ..config import PATH_DATA

class Task:
    def __init__(self, *, id: Hashable,  query: str, answer: str, properties: dict = None):
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
        self.data_path = self.data_dir / f'{self.bench_name}_{split}.jsonl'
        self.tasks: dict[Hashable, Task] = dict()
        self.template = template
        
    def load(self, *, indices: Iterable = None, download: bool = False):
        if not self.data_path.is_file() and download is not None: self.download()
        with open(self.data_path) as f:
            tasks_raw: list[dict] = [json.loads(line) for line in f if line.strip()]
            if indices is None:
                pairs = list(enumerate(tasks_raw))
            else:
                pairs = [(i, tasks_raw[i]) for i in indices]  # right now stores id as the index
            for id, task_raw in pairs:
                task = Task(id=id, query=task_raw[self.query_key], answer=task_raw[self.answer_key], properties=task_raw)
                self.tasks.update({task.id: task})
    
    def chat_prompt(self, task) -> str:
        raise NotImplementedError
    
    def grade(self, output: str, task_id: Hashable) -> tuple[int, str]:
        raise NotImplementedError
    
    def _get_task(self, task_id: Hashable):
        return self.tasks[task_id]
    
    def __iter__(self):
        return zip(self.tasks.keys(), self.tasks.values())
    
    def __len__(self):
        return len(self.tasks)
    
    def download(self) -> None:
        load_dataset(self.hf_path, self.hf_name, split=self.split).to_json(self.data_path)

