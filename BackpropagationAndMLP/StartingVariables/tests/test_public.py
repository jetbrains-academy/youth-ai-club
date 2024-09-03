import unittest

import torch.nn.functional as F
import torch
from task import hprebn_bp, embcat_bp, W1_bp, b1_bp, emb_bp, C_bp
from solution import correct_hprebn_bp, correct_embcat_bp, correct_W1_bp, correct_b1_bp, correct_emb_bp, correct_C_bp
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
        :param dbndiff: torch.Tensor[32, 64]
        :param dbnmeani: torch.Tensor[1, 64]
        :return: dhprebn: torch.Tensor[32, 64]
        '''
        maxdiff = 0
        name = "hprebn"
        atol = 1e-6
        n = 32
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            dbndiff = torch.randn((32, 64), generator=g)
            dbnmeani = torch.randn((1, 64), generator=g)

            student = timer(hprebn_bp)(n, dbndiff, dbnmeani)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_hprebn_bp)(n, dbndiff, dbnmeani)
            max_time_author = max(max_time_author, timer.get())

            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_embcat(self):
        '''
        :param dhprebn: torch.Tensor[32, 64]
        :param W1: torch.Tensor[30, 64]
        :return: dembcat: torch.Tensor[32, 30]
        '''
        maxdiff = 0
        name = "embcat"
        atol = 1e-4
        max_time_student = 0
        max_time_author = 0
        for i in range(3):
            g = torch.Generator().manual_seed(i + 1)
            dhprebn = torch.randn((32, 64), generator=g)
            W1 = torch.randn((30, 64), generator=g)

            student = timer(embcat_bp)(dhprebn, W1)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_embcat_bp)(dhprebn, W1)
            max_time_author = max(max_time_author, timer.get())
            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_W1(self):
        '''
        :param embcat: torch.Tensor[32, 30]
        :param dhprebn: torch.Tensor[32, 64]
        :return: dW1: torch.Tensor[30, 64]
        '''
        maxdiff = 0
        name = "W1"
        atol = 1e-6
        max_time_student = 0
        max_time_author = 0
        for i in range(3):
            g = torch.Generator().manual_seed(i + 1)
            embcat = torch.randn((32, 30), generator=g)
            dhprebn = torch.randn((32, 64), generator=g)

            student = timer(W1_bp)(embcat, dhprebn)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_W1_bp)(embcat, dhprebn)
            max_time_author = max(max_time_author, timer.get())

            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_b1(self):
        '''
        :param dhprebn: torch.Tensor[32, 64]
        :return: db1: torch.Tensor[64]
        '''
        maxdiff = 0
        name = "b1"
        atol = 1e-5
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            dhprebn = torch.randn((32, 64), generator=g)

            student = timer(b1_bp)(dhprebn)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_b1_bp)(dhprebn)
            max_time_author = max(max_time_author, timer.get())

            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_emb(self):
        '''
        :param dembcat: torch.Tensor[32, 30]
        :param emb: torch.Tensor[32, 3, 10]
        :return: demb: torch.Tensor[32, 3, 10]
        '''
        maxdiff = 0
        name = "emb"
        atol = 1e-6
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            dembcat = torch.randn((32, 30), generator=g)
            emb = torch.randn((32, 3, 10), generator=g)

            student = timer(emb_bp)(dembcat, emb)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_emb_bp)(dembcat, emb)
            max_time_author = max(max_time_author, timer.get())

            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_C(self):
        '''
        :param Xb: torch.Tensor[32, 3]
        :param demb: torch.Tensor[32, 3, 10]
        :param C: torch.Tensor[27, 10]
        :return: dC: torch.Tensor[27, 10]
        '''
        maxdiff = 0
        name = "C"
        atol = 1e-5
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            Xb = torch.randint(0, 27, (32, 3), generator=g)
            demb = torch.randn((32, 3, 10), generator=g)
            C = torch.randn((27, 10), generator=g)
            student = timer(C_bp)(Xb, demb, C)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_C_bp)(Xb, demb, C)
            max_time_author = max(max_time_author, timer.get())
            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')
