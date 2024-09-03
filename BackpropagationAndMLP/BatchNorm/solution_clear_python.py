import torch


def h_bp(dlogits, W2):
    # dlogits: torch.Tensor[32, 27], W2: torch.Tensor[64, 27]
    # dh: torch.Tensor[32, 64]
    dh = torch.zeros((32, 64))
    for i in range(dh.shape[0]):
        for j in range(dh.shape[1]):
            for k in range(dlogits.shape[1]):
                dh[i][j] += dlogits[i][k] * W2[j][k]

    return dh


def W2_bp(h, dlogits):
    # h: torch.Tensor[32, 64], dlogits: torch.Tensor[32, 27]
    # dW2: torch.Tensor[64, 27]
    dW2 = torch.zeros((64, 27))
    for i in range(h.shape[1]):
        for j in range(dlogits.shape[1]):
            for k in range(h.shape[0]):
                dW2[i][j] += h[k][i] * dlogits[k][j]
    return dW2


def b2_bp(dlogits):
    # dlogits: torch.Tensor[32, 27]
    # db2: torch.Tensor[27]
    db2 = torch.zeros(27)
    for i in range(db2.shape[0]):
        for j in range(dlogits.shape[0]):
            db2[i] += dlogits[j][i]
    return db2


def hpreact_bp(h, dh):
    # h: torch.Tensor[32, 64], dh: torch.Tensor[32, 64],
    # dhpreact: torch.Tensor[32, 64]
    dhpreact = torch.ones_like(h)
    for i in range(dhpreact.shape[0]):
        for j in range(dhpreact.shape[1]):
            dhpreact[i][j] = (1 - h[i][j] ** 2) * dh[i][j]
    return dhpreact


def bngain_bp(bnraw, dhpreact):
    # bnraw: torch.Tensor[32, 64], dhpreact: torch.Tensor[32, 64]
    # dbngain: torch.Tensor[1, 64]
    dbngain = torch.zeros((1, 64))
    for i in range(bnraw.shape[0]):
        for j in range(bnraw.shape[1]):
            dbngain[0][j] += bnraw[i][j] * dhpreact[i][j]
    return dbngain


def bnbias_bp(dhpreact):
    # dhpreact: torch.Tensor[32, 64]
    # dbnbias: torch.Tensor[1, 64]
    dbnbias = torch.zeros((1, 64))
    for i in range(dhpreact.shape[0]):
        for j in range(dhpreact.shape[1]):
            dbnbias[0][j] += dhpreact[i][j]
    return dbnbias


def bnraw_bp(dhpreact, bngain):
    # dhpreact: torch.Tensor[32, 64], bngain: torch.Tensor[1, 64]
    # dbnraw: torch.Tensor[32, 64]
    dbnraw = torch.zeros((32, 64))
    for i in range(dbnraw.shape[0]):
        for j in range(dbnraw.shape[1]):
            dbnraw[i][j] = dhpreact[i][j] * bngain[0][j]
    return dbnraw


def bnvar_inv_bp(bndiff, dbnraw):
    # bndiff: torch.Tensor[32, 64], dbnraw: torch.Tensor[32, 64]
    # dbnvar_inv: torch.Tensor[1, 64]
    dbnvar_inv = torch.zeros((1, 64))
    for i in range(bndiff.shape[0]):
        for j in range(bndiff.shape[1]):
            dbnvar_inv[0][j] += bndiff[i][j] * dbnraw[i][j]
    return dbnvar_inv


def bnvar_bp(bnvar, dbnvar_inv):
    # bnvar: torch.Tensor[1, 64], dbnvar_inv: torch.Tensor[1, 64]
    # dbnvar: torch.Tensor[1, 64]
    dbnvar = torch.zeros((1, 64))
    for i in range(bnvar.shape[1]):
        dbnvar[0][i] = -(2 ** -1) * (bnvar[0][i] + 1e-5) ** (-3 / 2) * dbnvar_inv[0][i]
    return dbnvar


def bndiff2_bp(bndiff2, n, dbnvar):
    # bndiff2: torch.Tensor[32, 64], n: int, dbnvar: torch.Tensor[1, 64]
    # dbndiff2: torch.Tensor[32, 64]
    dbndiff2 = None
    dbndiff2 = torch.ones_like(bndiff2) * (1 / (n - 1) * dbnvar)
    return dbndiff2


def bndiff_bp(bndiff, dbndiff2, bnvar_inv, dbnraw):
    # bndiff: torch.Tensor[32, 64], dbndiff2: torch.Tensor[32, 64],
    # bnvar_inv: torch.Tensor[1, 64], dbnraw: torch.Tensor[32, 64]
    # dbndiff: torch.Tensor[32, 64]
    dbndiff = torch.zeros_like(bndiff)
    for i in range(dbndiff.shape[0]):
        for j in range(dbndiff.shape[1]):
            dbndiff[i][j] = 2 * bndiff[i][j] * dbndiff2[i][j] + bnvar_inv[0][j] * dbnraw[i][j]
    return dbndiff


def bnmeani_bp(dbndiff):
    # dbndiff: torch.Tensor[32, 64]
    # dbnmeani: torch.Tensor[1, 64]
    dbnmeani = torch.zeros((1, 64))
    for i in range(dbndiff.shape[0]):
        for j in range(dbndiff.shape[1]):
            dbnmeani[0][j] -= dbndiff[i][j]
    return dbnmeani
