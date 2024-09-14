import random
import torch
import torch.nn.functional as F


def correct_hprebn_bp(n, bngain, bnvar_inv, bnraw, dhpreact):
    '''
    :param n: int
    :param bngain: torch.Tensor[1, 64]
    :param bnvar_inv: torch.Tensor[1, 64]
    :param bnraw: torch.Tensor[32, 64]
    :param dhpreact: torch.Tensor[32, 64]
    :return: dhprebn: torch.Tensor[32, 64]
    '''
    dhprebn = None
    dhprebn = bngain * bnvar_inv / n * (
            n * dhpreact - dhpreact.sum(0) - n / (n - 1) * bnraw * (dhpreact * bnraw).sum(0))
    return dhprebn