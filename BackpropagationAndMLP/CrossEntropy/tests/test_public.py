import unittest

import torch.nn.functional as F
import torch
from task import logits_bp
from solution import correct_logits_bp
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
    def test_logits(self):
        '''
        :param logits: torch.Tensor[32, 27]
        :param n: int
        :param Yb: torch.Tensor[32]
        :return: dlogits: torch.Tensor[32, 27]
        '''
        maxdiff = 0
        name = "logits"
        atol = 1e-9
        max_time_student = 0
        max_time_author = 0
        n = 32
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            logits = torch.randn((32, 27), generator=g)
            Yb = torch.randint(0, 27, (32,), generator=g)

            student = timer(logits_bp)(logits, n, Yb)
            max_time_student = max(timer.get(), max_time_student)

            answer = timer(correct_logits_bp)(logits, n, Yb)
            max_time_author = max(timer.get(), max_time_author)
