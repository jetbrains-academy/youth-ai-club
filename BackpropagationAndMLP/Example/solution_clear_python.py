import torch


def logprobs_bp(n, Yb, logprobs):
    # n: int, Yb: torch.Tensor[32], logprobs: torch.Tensor[32, 27]
    # dlogprobs: torch.Tensor[32, 27]
    dlogprobs = torch.zeros_like(logprobs)
    for i in range(n):
        dlogprobs[i][Yb[i]] = -1/n
    return dlogprobs


def probs_bp(probs, dlogprobs):
    # probs: torch.Tensor[32, 27], dlogprobs: torch.Tensor[32, 27]
    # dprobs: torch.Tensor[32, 27]
    dprobs = torch.zeros_like(probs)
    for i in range(dprobs.shape[0]):
        for j in range(dprobs.shape[1]):
            dprobs[i][j] = 1 / probs[i][j] * dlogprobs[i][j]

    return dprobs
