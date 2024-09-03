import torch


def correct_logprobs_bp(n, Yb, logprobs):
    '''
    :param n: int
    :param Yb: torch.Tensor[32]
    :param logprobs: torch.Tensor[32, 27]
    :return: dlogprobs: torch.Tensor[32, 27]
    '''
    dlogprobs = torch.zeros_like(logprobs)
    dlogprobs[range(n), Yb] = -1.0 / n
    return dlogprobs


def correct_probs_bp(probs, dlogprobs):
    '''
    :param probs: torch.Tensor[32, 27]
    :param dlogprobs: torch.Tensor[32, 27]
    :return: dp: torch.Tensor[32, 27]
    '''
    dprobs = 1.0 / probs * dlogprobs
    return dprobs
