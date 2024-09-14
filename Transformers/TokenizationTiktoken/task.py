import tiktoken
import torch

def tokenization(text: str):
    '''
    :param text: text to tokenize
    :return: encoded_text: list[int], stoi: dict[int, int], itos: dict[int, int]
    '''
    stoi, itos = {}, {}
    encoding = tiktoken.get_encoding("cl100k_base")
    encoded_text = encoding.encode(text)
    chars = sorted(list(set(encoded_text)))
    stoi = {s: i for i, s in enumerate(chars)}
    itos = {i: s for s, i in stoi.items()}
    encoded_text = [stoi[s] for s in encoded_text]
    return encoded_text, stoi, itos


def embeddings(vocab_size: int, n_embd: int, seed: int):
    '''
    :param vocab_size: number of different tokens
    :param n_embd: the dimension of the vector that represents the token
    :param seed: seed for random generator
    :return: C: torch.Tensor[vocab_size, n_embd] (C[i] is embedding vector for token i)
    '''
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((vocab_size, n_embd), generator=g)
    return C
