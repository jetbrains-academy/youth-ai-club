import unittest

import torch.nn.functional as F
import torch
from task import counts_sum_inv_bp, counts_sum_bp, counts_bp, norm_logits_bp, logit_maxes_bp, logits_bp
from BackpropagationAndMLP.time_measure import TimeMeasure, print_time

from solution import correct_counts_sum_inv_bp, correct_counts_sum_bp, correct_counts_bp, correct_norm_logits_bp, \
    correct_logit_maxes_bp, correct_logits_bp

timer = TimeMeasure()


def cmp(dt, t):
    if not isinstance(dt, torch.Tensor):
        return -1
    ex = torch.all(dt == t).item()
    app = torch.allclose(dt, t)
    maxdiff = (dt - t).abs().max().item()
    return ex, app, maxdiff


class TestCase(unittest.TestCase):
    def test_counts_sum_inv(self):
        '''
        :param counts: torch.Tensor[32, 27]
        :param dprobs: torch.Tensor[32, 27]
        :return: dcounts_sum_inv: torch.Tensor[32, 1]
        '''
        maxdiff = 0
        name = "counts_sum_inv"
        atol = 1e-5
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            counts = torch.randn((32, 27), generator=g)
            dprobs = torch.randn((32, 27), generator=g)

            student = timer(counts_sum_inv_bp)(counts, dprobs)
            max_time_student = max(timer.get(), max_time_student)

            answer = timer(correct_counts_sum_inv_bp)(counts, dprobs)
            max_time_author = max(timer.get(), max_time_author)

            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_counts_sum(self):
        '''
        :param counts_sum: torch.Tensor[32, 1]
        :param dcounts_sum_inv: torch.Tensor[32, 1]
        :return: dcouns_sum: torch.Tensor[32, 1]
        '''
        maxdiff = 0
        name = "counts_sum"
        atol = 1e-6
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            counts_sum = torch.randn((32, 1), generator=g)
            dcounts_sum_inv = torch.randn((32, 1), generator=g)

            student = timer(counts_sum_bp)(counts_sum, dcounts_sum_inv)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_counts_sum_bp)(counts_sum, dcounts_sum_inv)
            max_time_author = max(max_time_author, timer.get())

            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_counts(self):
        '''
        :param counts: torch.Tensor[32, 27]
        :param dcounts_sum: torch.Tensor[32, 1]
        :param counts_sum_inv: torch.Tensor[32, 1]
        :param dprobs: torch.Tensor[32, 27]
        :return: dcounts: torch.Tensor[32, 27]
        '''
        maxdiff = 0
        name = "counts"
        atol = 1e-6
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            counts = torch.randn((32, 27), generator=g)
            dcounts_sum = torch.randn((32, 1), generator=g)
            counts_sum_inv = torch.randn((32, 1), generator=g)
            dprobs = torch.randn((32, 27), generator=g)

            student = timer(counts_bp)(counts, dcounts_sum, counts_sum_inv, dprobs)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_counts_bp)(counts, dcounts_sum, counts_sum_inv, dprobs)
            max_time_author = max(max_time_author, timer.get())
            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_norm_logits(self):
        '''
        :param counts: torch.Tensor[32, 27]
        :param dcounts: torch.Tensor[32, 27]
        :return: dnorm_logits: torch.Tensor[32, 27]
        '''
        maxdiff = 0
        name = "norm_logits"
        atol = 1e-6
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            counts = torch.randn((32, 27), generator=g)
            dcounts = torch.randn((32, 27), generator=g)

            student = timer(norm_logits_bp)(counts, dcounts)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_norm_logits_bp)(counts, dcounts)
            max_time_author = max(max_time_author, timer.get())

            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_logit_maxes(self):
        '''
        :param dnorm_logits: torch.Tensor[32, 27]
        :return: dlogit_maxes: torch.Tensor[32, 1]
        '''
        maxdiff = 0
        name = "logit_maxes"
        atol = 1e-5
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            dnorm_logits = torch.randn((32, 27), generator=g)

            student = timer(logit_maxes_bp)(dnorm_logits)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_logit_maxes_bp)(dnorm_logits)
            max_time_author = max(max_time_author, timer.get())

            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_logits(self):
        '''
        :param logits: torch.Tensor[32, 27]
        :param dnorm_logits: torch.Tensor[32, 27]
        :param dlogit_maxes: torch.Tensor[32, 1]
        :return: dlogits: torch.Tensor[32, 27]
        '''
        maxdiff = 0
        name = "logits"
        atol = 1e-6
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            logits = torch.randn((32, 27), generator=g)
            dnorm_logits = torch.randn((32, 27), generator=g)
            dlogit_maxes = torch.randn((32, 1), generator=g)

            student = timer(logits_bp)(logits, dnorm_logits, dlogit_maxes)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_logits_bp)(logits, dnorm_logits, dlogit_maxes)
            max_time_author = max(max_time_author, timer.get())

            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')
