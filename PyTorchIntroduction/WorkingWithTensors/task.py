from typing import Tuple

import torch


def count_means(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return x.mean(1), x.mean(0)


def first_row_and_column(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return x[0, :], x[:, 0]


def create_chessboard(num_rows: int, num_cols: int) -> torch.Tensor:
    return (1 - torch.eye(2)).repeat((num_rows + 1) // 2, (num_cols + 1) // 2)[:num_rows, :num_cols].long()


def create_arithmetic_progressions(num_rows: int, num_cols: int) -> torch.Tensor:
    return (torch.arange(1, num_rows + 1)[:, None] * torch.arange(1, num_cols + 1)[None, :]).long()


def batch_flatten(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)
