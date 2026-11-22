import torch


class GRPOTrainer:
    def __init__(self, ):
        pass
    


# sequence level _logprobs，takes logprobs: masks to only output tokens, and sums
# (B, T-1), (B, ) -> (B, )
def seq_logprobs(logprobs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sum(mask*logprobs, dim=1)  # (B, )