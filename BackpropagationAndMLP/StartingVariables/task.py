import torch


def hprebn_bp(n, dbndiff, dbnmeani):
    '''
    :param n: int
    :param dbndiff: torch.Tensor[32, 64]
    :param dbnmeani: torch.Tensor[1, 64]
    :return: dhprebn: torch.Tensor[32, 64]
    '''
    dhprebn = None
    dhprebn = dbndiff + dbnmeani * (1 / n)
    return dhprebn


def embcat_bp(dhprebn, W1):
    '''
    :param dhprebn: torch.Tensor[32, 64]
    :param W1: torch.Tensor[30, 64]
    :return: dembcat: torch.Tensor[32, 30]
    '''
    dembcat = None
    dembcat = dhprebn @ W1.T
    return dembcat


def W1_bp(embcat, dhprebn):
    '''
    :param embcat: torch.Tensor[32, 30]
    :param dhprebn: torch.Tensor[32, 64]
    :return: dW1: torch.Tensor[30, 64]
    '''
    dW1 = None
    dW1 = embcat.T @ dhprebn
    return dW1


def b1_bp(dhprebn):
    '''
    :param dhprebn: torch.Tensor[32, 64]
    :return: db1: torch.Tensor[64]
    '''
    db1 = None
    db1 = dhprebn.sum(0)
    return db1


def emb_bp(dembcat, emb):
    '''
    :param dembcat: torch.Tensor[32, 30]
    :param emb: torch.Tensor[32, 3, 10]
    :return: demb: torch.Tensor[32, 3, 10]
    '''
    demb = None
    demb = dembcat.view(emb.shape)
    return demb


def C_bp(Xb, demb, C):
    '''
    :param Xb: torch.Tensor[32, 3]
    :param demb: torch.Tensor[32, 3, 10]
    :param C: torch.Tensor[27, 10]
    :return: dC: torch.Tensor[27, 10]
    '''
    dC = None
    dC = torch.zeros_like(C)
    for k in range(Xb.shape[0]):
        for j in range(Xb.shape[1]):
            ix = Xb[k, j]
            dC[ix] += demb[k, j]
    return dC
