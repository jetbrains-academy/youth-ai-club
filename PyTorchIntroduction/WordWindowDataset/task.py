from typing import List, Tuple, Set, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

corpus = [
    "We always come to Paris",
    "The professor is from Australia",
    "I live in Stanford",
    "He comes from Taiwan",
    "The capital of Turkey is Ankara",
]
locations = {"australia", "ankara", "paris", "stanford", "taiwan", "turkey"}

UNKNOWN_TOKEN = "<unk>"  # index 0
PADDING_TOKEN = "<pad>"  # index 1
WINDOW_SIZE = 2


def preprocess_sentence(sentence: str) -> List[str]:
    """Simple tokenizer: move to lowercase and split into words"""
    return sentence.lower().split()


def sentence2ids(sentence: str, word_to_idx: Dict[str, int]) -> List[int]:
    """
    Processes to list of tokens (words) with `preprocess sentence,
    then converts each token into idx with `word_to_idx`.
    `word_to_idx`: maps token to token_idx.
    If token not in `word_to_idx` - map it to 0 (UNKNOWN_TOKEN idx).
    """
    return [word_to_idx.get(word, 0) for word in preprocess_sentence(sentence)]


def build_word_to_idx(corpus: List[str]) -> Tuple[List[str], Dict[str, int]]:
    # Set of all words after `preprocess_sentence` of corpus sentences
    words_set = set()
    for sentence in corpus:
        words = preprocess_sentence(sentence)
        words_set = words_set | set(words)

    idx_to_word = [UNKNOWN_TOKEN, PADDING_TOKEN] + list(words_set)
    word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}
    return idx_to_word, word_to_idx


class LocationsDataset(Dataset):
    def __init__(self, corpus: List[str], locations: Set[str],
                 idx_to_word: Optional[List[str]] = None, word_to_idx: Optional[Dict[str, int]] = None):
        """
        Input:
            `corpus` - list of sentences
            `locations` - set of locations
        """
        super().__init__()
        self.corpus_ = corpus
        self.locations_ = locations

        if idx_to_word is None or word_to_idx is None:
            self.idx_to_word_, self.word_to_idx_ = build_word_to_idx(corpus)
        else:
            self.idx_to_word_, self.word_to_idx_ = idx_to_word, word_to_idx

    def __len__(self) -> int:
        return len(self.corpus_)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        words = preprocess_sentence(self.corpus_[idx])
        word_idx_list = [self.word_to_idx_.get(word, 0) for word in words]
        labels = [int(word in self.locations_) for word in words]
        return word_idx_list, labels


def collate_fn_sentence(data: List[Tuple[List[str], List[bool]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
    """
    Input:
        * data - list of dataset's __getitem__ outputs
    Return:
        * word_idx_padded: Tensor of len(data) x padded_length - padded indices of words from data[:, 0]
        * labels: Tensor of len(data) x padded_length - padded labels from data[:, 1]
        * lengths: LongTensor of len(data) - lengths (number of tokens) of each output from data
    Usage:
        ids = [0, 1, 2, 3]
        data = [dataset[idx] for idx in ids]
        batch = collate_fn_sentence(data)
    """
    word_idx_padded_list, labels = [], []
    for word_idx_list, labels_ in data:
        window = [1] * WINDOW_SIZE # 1 - padding_token_idx
        word_idx_padded_list.append(torch.LongTensor(window + word_idx_list + window))
        labels.append(torch.LongTensor(labels_))

    lengths = torch.LongTensor([len(labels_) for labels_ in labels])

    """
        Pad each tensor of `word_idx_padded_list` to align all lengths. Use `nn.utils.rnn.pad_sequence`.
        Pad with 1 - index of `PADDING_TOKEN`
    """
    word_idx_padded = nn.utils.rnn.pad_sequence(
        word_idx_padded_list, batch_first=True, padding_value=1,
    )
    """
        Pad each tensor of `labels` to align all lengths. Use `nn.utils.rnn.pad_sequence`.
        Pad with 0 - `locations` doesn't contain `PADDING_TOKEN`.  
    """
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return word_idx_padded, labels, lengths


BATCH_SIZE = 2

dataset = LocationsDataset(corpus, locations)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_sentence)
