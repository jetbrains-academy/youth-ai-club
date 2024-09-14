import torch
import torch.nn.functional as F


def counts_sum_inv_bp(counts, dprobs):
    # counts: torch.Tensor[32, 27], dprobs: torch.Tensor[32, 27]
    # dcounts_sum_inv: torch.Tensor[32, 1]
    dcounts_sum_inv = torch.zeros((32, 1))
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            dcounts_sum_inv[i] += counts[i][j] * dprobs[i][j]
    return dcounts_sum_inv


def counts_sum_bp(counts_sum, dcounts_sum_inv):
    # counts_sum: torch.Tensor[32, 1], dcounts_sum_inv: torch.Tensor[32, 1]
    # dcounts_sum: torch.Tensor[32, 1]
    dcounts_sum = torch.zeros((32, 1))
    for i in range(dcounts_sum.shape[0]):
        dcounts_sum[i] = -(counts_sum[i] ** -2) * dcounts_sum_inv[i]
    return dcounts_sum


def counts_bp(counts, dcounts_sum, counts_sum_inv, dprobs):
    # counts: torch.Tensor[32, 27], dcounts_sum: torch.Tensor[32, 1],
    # counts_sum_inv: torch.Tensor[32, 1], dprobs: torch.Tensor[32, 27]
    # dcounts: torch.Tensor[32, 27]
    dcounts = torch.ones_like(counts)
    for i in range(dcounts.shape[0]):
        for j in range(dcounts.shape[1]):
            dcounts[i][j] = dcounts_sum[i] + counts_sum_inv[i] * dprobs[i][j]

    return dcounts


def norm_logits_bp(counts, dcounts):
    # counts: torch.Tensor[32, 27], dcounts: torch.Tensor[32, 27]
    # dnorm_logits: torch.Tensor[32, 27]
    dnorm_logits = torch.ones_like(counts)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            dnorm_logits[i][j] = counts[i][j] * dcounts[i][j]
    return dnorm_logits


def logit_maxes_bp(dnorm_logits):
    # dnorm_logits: torch.Tensor[32, 27]
    # dlogit_maxes: torch.Tensor[32, 1]
    dlogit_maxes = torch.zeros((32, 1))
    for i in range(dnorm_logits.shape[0]):
        for j in range(dnorm_logits.shape[1]):
            dlogit_maxes[i] -= dnorm_logits[i][j]
    return dlogit_maxes


def logits_bp(logits, dnorm_logits, dlogit_maxes):
    # logits: torch.Tensor[32, 27], dnorm_logits: torch.Tensor[32, 27], dlogit_maxes: torch.Tensor[32, 1]
    # dlogits: torch.Tensor[32, 27]
    dlogits = torch.zeros_like(logits)
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            dlogits[i][j] += dnorm_logits[i][j]
            if logits[i][j] == logits[i].max():
                dlogits[i][j] += dlogit_maxes[i][0]

    return dlogits
