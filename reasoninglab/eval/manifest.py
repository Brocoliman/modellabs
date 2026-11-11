from collections.abc import Iterable
from pathlib import Path
from uuid import UUID, uuid4


class Task:
    def __init__(self, *, idx: int, problem: str, answer: str, properties: dict):
        self.id = uuid4()
        self.idx = idx
        self.problem = problem
        self.answer = answer
        self.properties = properties


class TaskManifest:
    data_path: Path
    
    def __init__(self, template: str):
        self.tasks: dict[UUID, Task] = dict()
        self.template = template
        
    def load(self, indices: Iterable = None):
        pass
    
    def chat_prompt(self, task) -> str:
        pass
    
    def grade(self, output: str, task_id: UUID) -> tuple[int, str]:
        pass
    
    def _get_task(self, task_id: UUID):
        return self.tasks[task_id]
    
    def __iter__(self):
        return zip(self.tasks.keys(), self.tasks.values())
    
    def __len__(self):
        return len(self.tasks)
    
    @staticmethod
    def download() -> None:
        pass

        