import torch


def hprebn_bp(n, dbndiff, dbnmeani):
    # n: int, dbndiff: torch.Tensor[32, 64], dbnmeani: torch.Tensor[1, 64]
    # dhprebn: torch.Tensor[32, 64]
    dhprebn = torch.zeros((32, 64))
    for i in range(dhprebn.shape[0]):
        for j in range(dhprebn.shape[1]):
            dhprebn[i][j] = dbndiff[i][j] + dbnmeani[0][j] * 1 / n

    return dhprebn


def embcat_bp(dhprebn, W1):
    # dhprebn: torch.Tensor[32, 64], W1: torch.Tensor[30, 64]
    # dembcat: torch.Tensor[32, 30]
    dembcat = torch.zeros((32, 30))
    for i in range(dhprebn.shape[0]):
        for j in range(W1.shape[0]):
            for k in range(W1.shape[1]):
                dembcat[i][j] += dhprebn[i][k] * W1[j][k]
    return dembcat


def W1_bp(embcat, dhprebn):
    # embcat: torch.Tensor[32, 30], dhprebn: torch.Tensor[32, 64]
    # dW1: torch.Tensor[30, 64]
    dW1 = torch.zeros((30, 64))
    for i in range(embcat.shape[1]):
        for j in range(dhprebn.shape[1]):
            for k in range(embcat.shape[0]):
                dW1[i][j] += embcat[k][i] * dhprebn[k][j]

    return dW1


def b1_bp(dhprebn):
    # dhprebn: torch.Tensor[32, 64]
    # db1: torch.Tensor[64]
    db1 = torch.zeros(64)
    for i in range(dhprebn.shape[0]):
        for j in range(dhprebn.shape[1]):
            db1[j] += dhprebn[i][j]
    return db1


def emb_bp(dembcat, emb):
    # dembcat: torch.Tensor[32, 30], emb: torch.Tensor[32, 3, 10]
    # demb: torch.Tensor[32, 3, 10]
    demb = None
    demb = dembcat.view(emb.shape)
    return demb


def C_bp(Xb, demb, C):
    # Xb: torch.Tensor[32, 3], demb: torch.Tensor[32, 3, 10], C: torch.Tensor[27, 10]
    # dC: torch.Tensor[27, 10]
    dC = None
    dC = torch.zeros_like(C)
    for k in range(Xb.shape[0]):
        for j in range(Xb.shape[1]):
            ix = Xb[k][j]
            for i in range(demb.shape[2]):
                dC[ix][i] += demb[k][j][i]
    return dC
