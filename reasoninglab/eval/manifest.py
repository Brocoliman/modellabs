from collections.abc import Iterable, Hashable
from pathlib import Path


class Task:
    def __init__(self, *, id: Hashable,  query: str, answer: str, properties: dict = None):
        self.id = id
        self.query = query
        self.answer = answer
        self.properties = properties


class TaskManifest:
    data_path: Path
    
    def __init__(self, template: str):
        self.tasks: dict[Hashable, Task] = dict()
        self.template = template
        
    def load(self, indices: Iterable = None):
        raise NotImplementedError
    
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
    
    @staticmethod
    def download() -> None:
        raise NotImplementedError

