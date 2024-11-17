import torch
import unittest

from task import (
    count_means,
    first_row_and_column,
    create_chessboard,
    create_arithmetic_progressions,
    batch_flatten,
)


class TestCase(unittest.TestCase):
    def test_count_means(self):
        x = torch.tensor([
            [1, 2.2, 9.6],
            [4, -7.2, 6.3],
        ])
        row_means = torch.tensor([12.8 / 3, 3.1 / 3])
        col_means = torch.tensor([2.5, -2.5, 15.9 / 2])
        pred_row_means, pred_col_means = count_means(x)
        self.assertTrue(torch.allclose(pred_row_means, row_means), msg="Row means: x = [[1, 2.2, 9.6], [4, -7.2, 6.3]]")
        self.assertTrue(torch.allclose(pred_col_means, col_means), msg="Col means: x = [[1, 2.2, 9.6], [4, -7.2, 6.3]]")

    def test_count_means_for_randn(self):
        x = torch.randn((20, 10))
        row_means = x.mean(1)
        col_means = x.mean(0)
        pred_row_means, pred_col_means = count_means(x)
        self.assertTrue(torch.allclose(pred_row_means, row_means), msg="Row means: x = torch.randn(20, 10)")
        self.assertTrue(torch.allclose(pred_col_means, col_means), msg="Col means: x = torch.randn(20, 10)")

    def test_first_row_and_column(self):
        x = torch.tensor([
            [1, 2.2, 9.6],
            [4, -7.2, 6.3],
        ])
        first_row = torch.tensor([1, 2.2, 9.6])
        first_col = torch.tensor([1., 4.])
        pred_first_row, pred_first_col = first_row_and_column(x)
        self.assertTrue(torch.allclose(pred_first_row, first_row), msg="First row: x = [[1, 2.2, 9.6], [4, -7.2, 6.3]]")
        self.assertTrue(torch.allclose(pred_first_col, first_col), msg="First col: x = [[1, 2.2, 9.6], [4, -7.2, 6.3]]")

    def test_first_row_and_column_for_randn(self):
        x = torch.randn((20, 10))
        pred_first_row, pred_first_col = first_row_and_column(x)
        self.assertTrue(torch.allclose(pred_first_row, x[0, :]), msg="First row: x = torch.randn(20, 10)")
        self.assertTrue(torch.allclose(pred_first_col, x[:, 0]), msg="First col: x = torch.randn(20, 10)")

    def test_chessboard(self):
        x = create_chessboard(5, 4)
        answer = torch.tensor([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ])
        self.assertTrue(torch.allclose(x, answer), msg="Chessboard 5x4")

    def test_arithmetic_progressions(self):
        x = create_arithmetic_progressions(5, 4)
        answer = torch.tensor([
            [1, 2, 3, 4],
            [2, 4, 6, 8],
            [3, 6, 9, 12],
            [4, 8, 12, 16],
            [5, 10, 15, 20],
        ])
        self.assertTrue(torch.allclose(x, answer), msg="arithmetic_progressions 5x4")

    def test_batch_flatten(self):
        x = torch.randn((3, 4, 5, 6))
        flatten = torch.flatten(x, start_dim=1)
        pred_flatten = batch_flatten(x)
        self.assertTrue(torch.allclose(pred_flatten, flatten), msg="x = torch.randn(3, 4, 5, 6)")
