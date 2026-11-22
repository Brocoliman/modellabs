import torch


class GRPOTrainer:
    def __init__(self, ):
        pass
    


# sequence level _logprobs，takes logprobs: masks and sums
# (B, T-1), (B, ) -> (B, )
# assumption: 
#   tokens[T_in-1:T_in-1+lengths[b]] is the model output, which includes EOS
#   -1 comes from the left shift of logprobs
def seq_logprobs(logprobs: torch.Tensor, lengths: torch.Tensor, T_in: int, ) -> torch.Tensor:
    B, T_ = logprobs.shape
    mask = torch.zeros_like(logprobs)  # (B, T-1)
    batch_idx = torch.arange(B, device=logprobs.device)
    mask[batch_idx, T_in-1:T_in+1+lengths] = 1
    