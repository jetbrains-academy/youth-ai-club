import torch


def correct_h_bp(dlogits, W2):
    '''
    :param dlogits: torch.Tensor[32, 27]
    :param W2: torch.Tensor[64, 27]
    :return: torch.Tensor[32, 64]
    '''
    dh = None
    dh = dlogits @ W2.T
    return dh


def correct_W2_bp(h, dlogits):
    '''
    :param h: torch.Tensor[32, 64]
    :param dlogits: torch.Tensor[32, 27]
    :return: dW2: torch.Tensor[64, 27]
    '''
    dW2 = None
    dW2 = h.T @ dlogits
    return dW2


def correct_b2_bp(dlogits):
    '''
    :param dlogits: torch.Tensor[32, 27]
    :return: db2: torch.Tensor[27]
    '''
    db2 = None
    db2 = dlogits.sum(0)
    return db2


def correct_hpreact_bp(h, dh):
    '''
    :param h: torch.Tensor[32, 64]
    :param dh: torch.Tensor[32, 64]
    :return: dhpreact: torch.Tensor[32, 64]
    '''
    dhpreact = None
    dhpreact = (1.0 - h ** 2) * dh
    return dhpreact


def correct_bngain_bp(bnraw, dhpreact):
    '''
    :param bnraw: torch.Tensor[32, 64]
    :param dhpreact: torch.Tensor[32, 64]
    :return: dbngain: torch.Tensor[1, 64]
    '''
    dbngain = None
    dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    return dbngain


def correct_bnbias_bp(dhpreact):
    '''
    :param dhpreact: torch.Tensor[32, 64]
    :return: dbnbias: torch.Tensor[1, 64]
    '''
    dbnbias = None
    dbnbias = dhpreact.sum(0)
    return dbnbias


def correct_bnraw_bp(dhpreact, bngain):
    '''
    :param dhpreact: torch.Tensor[32, 64]
    :param bngain: torch.Tensor[1, 64]
    :return: dbnraw: torch.Tensor[32, 64]
    '''
    dbnraw = None
    dbnraw = dhpreact * bngain
    return dbnraw


def correct_bnvar_inv_bp(bndiff, dbnraw):
    '''
    :param bndiff: torch.Tensor[32, 64]
    :param dbnraw: torch.Tensor[32, 64]
    :return: dbnvar_inv: torch.Tensor[1, 64]
    '''
    dbnvar_inv = None
    dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
    return dbnvar_inv


def correct_bnvar_bp(bnvar, dbnvar_inv):
    '''
    :param bnvar: torch.Tensor[1, 64]
    :param dbnvar_inv: torch.Tensor[1, 64]
    :return: dbnvar: torch.Tensor[1, 64]
    '''
    dbnvar = None
    dbnvar = -(2 ** -1) * (bnvar + 1e-5) ** (-3 / 2) * dbnvar_inv
    return dbnvar


def correct_bndiff2_bp(bndiff2, n, dbnvar):
    '''
    :param bndiff2: torch.Tensor[32, 64]
    :param n: int
    :param dbnvar: torch.Tensor[1, 64]
    :return: dbndiff2: torch.Tensor[32, 64]
    '''
    dbndiff2 = None
    dbndiff2 = torch.ones_like(bndiff2) * (1 / (n - 1) * dbnvar)
    return dbndiff2


def correct_bndiff_bp(bndiff, dbndiff2, bnvar_inv, dbnraw):
    '''
    :param bndiff: torch.Tensor[32, 64]
    :param dbndiff2: torch.Tensor[32, 64]
    :param bnvar_inv: torch.Tensor[1, 64]
    :param dbnraw: torch.Tensor[32, 64]
    :return: dbndiff: torch.Tensor[32, 64]
    '''
    dbndiff = None
    dbndiff = 2 * bndiff * dbndiff2 + bnvar_inv * dbnraw
    return dbndiff


def correct_bnmeani_bp(dbndiff):
    '''
    :param dbndiff: torch.Tensor[32, 64]
    :return: dbnmeani: torch.Tensor[1, 64]
    '''
    dbnmeani = None
    dbnmeani = -dbndiff.sum(0, keepdim=True)
    return dbnmeani
