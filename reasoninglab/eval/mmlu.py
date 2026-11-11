from datasets import load_dataset
from dotenv import load_dotenv
from uuid import UUID

from .manifest import TaskManifest, Task
from ..config import *

load_dotenv()

# ===== Prompts


# =====

class MMLUManifest(TaskManifest):
    data_path = Path(PATH_MMLU)
    
    def __init__(self, template):
        super().__init__(template)
        assert self.template.find('{task}') != -1, 'Could not find {task} in prompt template'
    
    def chat_prompt(self, task):
        return self.template.format(task=task)

    @staticmethod
    def download():
        hf_dataset = load_dataset("cais/mmlu", "all", split="dev")