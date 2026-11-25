import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import *
from ..infer import ChatModel
from ..eval.manifest import TaskManifest
from ..eval.runner import Runner


class GRPOTrainer:
    """Trainer class for RLVR
    
    Requires:
        chatmodel: ChatModel
        tasks: TaskManifest
            use for generate prompt (chat_prompt) and get reward (score)
        runner: Runner (optional)
            useful for periodic eval, if nothing is passed, periodic eval does not happen
    """
    def __init__(self, chatmodel: ChatModel, *, tasks: TaskManifest, train: RLConfig, runner: Runner = None):
        self.chatmodel = chatmodel
        self.model = self.chatmodel.model
        self.tasks = tasks
        self.runner = runner
    


# sequence level _logprobs，takes logprobs: masks to only output tokens, and sums
# (B, T-1), (B, ) -> (B, )
def seq_logprobs(logprobs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sum(mask*logprobs, dim=1)