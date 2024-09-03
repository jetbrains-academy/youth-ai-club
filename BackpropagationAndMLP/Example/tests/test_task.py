import unittest

import torch
from task import logprobs_bp, probs_bp
from solution import correct_logprobs_bp, correct_probs_bp
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

    def test_logprobs(self):
        '''
        :param n: int
        :param Yb: torch.Tensor[32]
        :param logprobs: torch.Tensor[32, 27]
        :return: dlogprobs: torch.Tensor[32, 27]
        '''
        n = 32
        maxdiff = 0
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            Yb = torch.randint(0, 27, (32,), generator=g)
            logprobs = torch.randn((32, 27), generator=g)
            student = timer(logprobs_bp)(n, Yb, logprobs)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_logprobs_bp)(n, Yb, logprobs)
            max_time_author = max(max_time_author, timer.get())
            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'counts_sum_inv: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        atol = 1e-6
        self.assertLess(maxdiff, atol, msg=f"logprobs: your maxdiff is more then {atol}, {maxdiff}")
        print_time('logprobs', max_time_author, 'author')
        print_time('logprobs', max_time_student, 'student')

    def test_probs(self):
        '''
        :param probs: torch.Tensor[32, 27]
        :param dlogprobs: torch.Tensor[32, 27]
        :return: dp: torch.Tensor[32, 27]
        '''
        maxdiff = 0
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            probs = torch.rand((32, 37), generator=g)
            dlogprobs = torch.rand((32, 37), generator=g)
            student = timer(probs_bp)(probs, dlogprobs)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_probs_bp)(probs, dlogprobs)
            max_time_author = max(max_time_author, timer.get())
            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'counts_sum_inv: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        atol = 1e-6
        self.assertLess(maxdiff, atol, msg=f"probs: your maxdiff is more then {atol}, {maxdiff}")
        print_time('probs', max_time_author, 'author')
        print_time('probs', max_time_student, 'student')
