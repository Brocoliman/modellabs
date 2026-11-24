import torch
import torch.nn as nn
import torch.nn.functional as F

from ..infer import ChatModel


class GRPOTrainer:
    def __init__(self, chatmodel: ChatModel):
        self.chatmodel = chatmodel
        self.model = self.chatmodel.model
    


# sequence level _logprobs，takes logprobs: masks to only output tokens, and sums
# (B, T-1), (B, ) -> (B, )
def seq_logprobs(logprobs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sum(mask*logprobs, dim=1)