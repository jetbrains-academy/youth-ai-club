import torch


def correct_tokenization(words: list):
    '''
    :param words: list of words
    :return: stoi: dict[str, int], itos: dict[int, str]
    '''
    stoi, itos = {}, {}
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['#'] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def correct_embeddings(vocab_size: int, n_embd: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((vocab_size, n_embd), generator=g)
    return C
