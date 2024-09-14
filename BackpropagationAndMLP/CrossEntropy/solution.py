import torch
import torch.nn.functional as F


def correct_logits_bp(logits, n, Yb):
    '''
    :param logits: torch.Tensor[32, 27]
    :param n: int
    :param Yb: torch.Tensor[32]
    :return: dlogits: torch.Tensor[32, 27]
    '''
    dlogits = F.softmax(logits, 1)
    dlogits[range(n), Yb] -= 1
    dlogits /= n
    return dlogits
