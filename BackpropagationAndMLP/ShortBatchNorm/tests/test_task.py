import unittest

import torch.nn.functional as F
import torch
from task import hprebn_bp
from solution import correct_hprebn_bp
from BackpropagationAndMLP.time_measure import TimeMeasure, print_time

timer = TimeMeasure()


def cmp(dt, t):
    if not isinstance(dt, torch.Tensor):
        return -1
    ex = torch.all(dt == t).item()
    app = torch.allclose(dt, t)
    maxdiff = (dt - t).abs().max().item()
    return ex, app, maxdiff


class TestCase(unittest.TestCase):
    def test_hprebn(self):
        '''
        :param n: int
        :param bngain: torch.Tensor[1, 64]
        :param bnvar_inv: torch.Tensor[1, 64]
        :param bnraw: torch.Tensor[32, 64]
        :param dhpreact: torch.Tensor[32, 64]
        :return: dhprebn: torch.Tensor[32, 64]
        '''
        maxdiff = 0
        name = "hprebn"
        atol = 1e-9
        max_time_student = 0
        max_time_author = 0
        n = 32
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            bngain = torch.randn((1, 64), generator=g)
            bnvar_inv = torch.randn((1, 64), generator=g)
            bnraw = torch.randn((32, 64), generator=g)
            dhpreact = torch.randn((32, 64), generator=g)

            student = timer(hprebn_bp)(n, bngain, bnvar_inv, bnraw, dhpreact)
            max_time_student = max(timer.get(), max_time_student)

            answer = timer(correct_hprebn_bp)(n, bngain, bnvar_inv, bnraw, dhpreact)
            max_time_author = max(timer.get(), max_time_author)
