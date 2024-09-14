import torch
import torch.nn.functional as F


def correct_counts_sum_inv_bp(counts, dprobs):
    '''
    :param counts: torch.Tensor[32, 27]
    :param dprobs: torch.Tensor[32, 27]
    :return: dcounts_sum_inv: torch.Tensor[32, 1]
    '''
    dcounts_sum_inv = None
    dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
    return dcounts_sum_inv


def correct_counts_sum_bp(counts_sum, dcounts_sum_inv):
    '''
    :param counts_sum: torch.Tensor[32, 1]
    :param dcounts_sum_inv: torch.Tensor[32, 1]
    :return: dcouns_sum: torch.Tensor[32, 1]
    '''
    dcounts_sum = None
    dcounts_sum = -(counts_sum ** -2) * dcounts_sum_inv
    return dcounts_sum


def correct_counts_bp(counts, dcounts_sum, counts_sum_inv, dprobs):
    '''
    :param counts: torch.Tensor[32, 27]
    :param dcounts_sum: torch.Tensor[32, 1]
    :param counts_sum_inv: torch.Tensor[32, 1]
    :param dprobs: torch.Tensor[32, 27]
    :return: dcounts: torch.Tensor[32, 27]
    '''
    dcounts = None
    dcounts = torch.ones_like(counts) * dcounts_sum + counts_sum_inv * dprobs
    return dcounts


def correct_norm_logits_bp(counts, dcounts):
    '''
    :param counts: torch.Tensor[32, 27]
    :param dcounts: torch.Tensor[32, 27]
    :return: dnorm_logits: torch.Tensor[32, 27]
    '''
    dnorm_logits = None
    dnorm_logits = counts * dcounts
    return dnorm_logits


def correct_logit_maxes_bp(dnorm_logits):
    '''
    :param dnorm_logits: torch.Tensor[32, 27]
    :return: dlogit_maxes: torch.Tensor[32, 1]
    '''
    dlogit_maxes = None
    dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
    return dlogit_maxes


def correct_logits_bp(logits, dnorm_logits, dlogit_maxes):
    '''
    :param logits: torch.Tensor[32, 27]
    :param dnorm_logits: torch.Tensor[32, 27]
    :param dlogit_maxes: torch.Tensor[32, 1]
    :return: dlogits: torch.Tensor[32, 27]
    '''
    dlogits = None
    dlogits = torch.ones_like(logits) * dnorm_logits + F.one_hot(logits.max(1).indices,
                                                                 num_classes=logits.shape[1]) * dlogit_maxes
    return dlogits
