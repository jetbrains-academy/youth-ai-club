import unittest

import torch.nn.functional as F
import torch
from task import h_bp, W2_bp, b2_bp, hpreact_bp, bngain_bp, bnbias_bp, bnraw_bp, bnvar_inv_bp, bnvar_bp, bndiff2_bp, \
    bndiff_bp, bnmeani_bp
from solution import correct_h_bp, correct_W2_bp, correct_b2_bp, correct_hpreact_bp, correct_bngain_bp, \
    correct_bnbias_bp, correct_bnraw_bp, correct_bnvar_inv_bp, correct_bnvar_bp, correct_bndiff2_bp, \
    correct_bndiff_bp, correct_bnmeani_bp

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
    def test_h(self):
        '''
        :param dlogits: torch.Tensor[32, 27]
        :param W2: torch.Tensor[64, 27]
        :return: torch.Tensor[32, 64]
        '''
        maxdiff = 0
        name = "h"
        atol = 1e-5
        max_time_student = 0
        max_time_author = 0
        for i in range(2):
            g = torch.Generator().manual_seed(i + 1)
            dlogits = torch.randn((32, 27), generator=g)
            W2 = torch.randn((64, 27), generator=g)

            student = timer(h_bp)(dlogits, W2)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_h_bp)(dlogits, W2)
            max_time_author = max(max_time_author, timer.get())

            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_W2(self):
        '''
        :param h: torch.Tensor[32, 64]
        :param dlogits: torch.Tensor[32, 27]
        :return: dW2: torch.Tensor[64, 27]
        '''
        maxdiff = 0
        name = "W2"
        atol = 1e-6
        max_time_student = 0
        max_time_author = 0
        for i in range(2):
            g = torch.Generator().manual_seed(i + 1)
            h = torch.randn((32, 64), generator=g)
            dlogits = torch.randn((32, 27), generator=g)

            student = timer(W2_bp)(h, dlogits)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_W2_bp)(h, dlogits)
            max_time_author = max(max_time_author, timer.get())

            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_b2(self):
        '''
        :param dlogits: torch.Tensor[32, 27]
        :return: db2: torch.Tensor[27]
        '''
        maxdiff = 0
        name = "b2"
        atol = 1e-5
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            dlogits = torch.randn((32, 27), generator=g)

            student = timer(b2_bp)(dlogits)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_b2_bp)(dlogits)
            max_time_author = max(max_time_author, timer.get())
            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_hpreact(self):
        '''
        :param h: torch.Tensor[32, 64]
        :param dh: torch.Tensor[32, 64]
        :return: dhpreact: torch.Tensor[32, 64]
        '''
        maxdiff = 0
        name = "hpreact"
        atol = 1e-6
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            h = torch.randn((32, 64), generator=g)
            dh = torch.randn((32, 64), generator=g)

            student = timer(hpreact_bp)(h, dh)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_hpreact_bp)(h, dh)
            max_time_author = max(max_time_author, timer.get())

            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_bngain(self):
        '''
        :param bnraw: torch.Tensor[32, 64]
        :param dhpreact: torch.Tensor[32, 64]
        :return: dbngain: torch.Tensor[1, 64]
        '''
        maxdiff = 0
        name = "bngain"
        atol = 1e-5
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            bnraw = torch.randn((32, 64), generator=g)
            dhpreact = torch.randn((32, 64), generator=g)

            student = timer(bngain_bp)(bnraw, dhpreact)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_bngain_bp)(bnraw, dhpreact)
            max_time_author = max(max_time_author, timer.get())

            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_bnbias(self):
        '''
        :param dhpreact: torch.Tensor[32, 64]
        :return: dbnbias: torch.Tensor[1, 64]
        '''
        maxdiff = 0
        name = "bnbias"
        atol = 1e-5
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            dhpreact = torch.randn((32, 64), generator=g)

            student = timer(bnbias_bp)(dhpreact)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_bnbias_bp)(dhpreact)
            max_time_author = max(max_time_author, timer.get())

            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_bnraw(self):
        '''
        :param dhpreact: torch.Tensor[32, 64]
        :param bngain: torch.Tensor[1, 64]
        :return: dbnraw: torch.Tensor[32, 64]
        '''
        maxdiff = 0
        name = "bnraw"
        atol = 1e-6
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            dhpreact = torch.randn((32, 64), generator=g)
            bngain = torch.randn((1, 64), generator=g)

            student = timer(bnraw_bp)(dhpreact, bngain)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_bnraw_bp)(dhpreact, bngain)
            max_time_author = max(max_time_author, timer.get())

            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_bnvar_inv(self):
        '''
        :param bndiff: torch.Tensor[32, 64]
        :param dbnraw: torch.Tensor[32, 64]
        :return: dbnvar_inv: torch.Tensor[1, 64]
        '''
        maxdiff = 0
        name = "bnvar_inv"
        atol = 1e-5
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            bndiff = torch.randn((32, 64), generator=g)
            dbnraw = torch.randn((32, 64), generator=g)

            student = timer(bnvar_inv_bp)(bndiff, dbnraw)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_bnvar_inv_bp)(bndiff, dbnraw)
            max_time_author = max(max_time_author, timer.get())
            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_bnvar(self):
        '''
        :param bnvar: torch.Tensor[1, 64]
        :param dbnvar_inv: torch.Tensor[1, 64]
        :return: dbnvar: torch.Tensor[1, 64]
        '''
        maxdiff = 0
        name = "bnvar"
        atol = 1e-6
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            bnvar = torch.randn((1, 64), generator=g)
            dbnvar_inv = torch.randn((1, 64), generator=g)
            student = timer(bnvar_bp)(bnvar, dbnvar_inv)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_bnvar_bp)(bnvar, dbnvar_inv)
            max_time_author = max(max_time_author, timer.get())
            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_bndiff2(self):
        '''
        :param bndiff2: torch.Tensor[32, 64]
        :param n: int
        :param dbnvar: torch.Tensor[1, 64]
        :return: dbndiff2: torch.Tensor[32, 64]
        '''
        maxdiff = 0
        name = "bndiff2"
        atol = 1e-6
        n = 32
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            bndiff2 = torch.randn((32, 64), generator=g)
            dbnvar = torch.randn((1, 64), generator=g)

            student = timer(bndiff2_bp)(bndiff2, n, dbnvar)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_bndiff2_bp)(bndiff2, n, dbnvar)
            max_time_author = max(max_time_author, timer.get())

            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_bndiff(self):
        '''
        :param bndiff: torch.Tensor[32, 64]
        :param dbndiff2: torch.Tensor[32, 64]
        :param bnvar_inv: torch.Tensor[1, 64]
        :param dbnraw: torch.Tensor[32, 64]
        :return: dbndiff: torch.Tensor[32, 64]
        '''
        maxdiff = 0
        name = "bndiff"
        atol = 1e-6
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            bndiff = torch.randn((32, 64), generator=g)
            dbndiff2 = torch.randn((32, 64), generator=g)
            bnvar_inv = torch.randn((1, 64), generator=g)
            dbnraw = torch.randn((32, 64), generator=g)

            student = timer(bndiff_bp)(bndiff, dbndiff2, bnvar_inv, dbnraw)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_bndiff_bp)(bndiff, dbndiff2, bnvar_inv, dbnraw)
            max_time_author = max(max_time_author, timer.get())

            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')

    def test_bnmeani(self):
        '''
        :param dbndiff: torch.Tensor[32, 64]
        :return: dbnmeani: torch.Tensor[1, 64]
        '''
        maxdiff = 0
        name = "bnmeani"
        atol = 1e-5
        max_time_student = 0
        max_time_author = 0
        for i in range(10):
            g = torch.Generator().manual_seed(i + 1)
            dbndiff = torch.randn((32, 64), generator=g)

            student = timer(bnmeani_bp)(dbndiff)
            max_time_student = max(max_time_student, timer.get())
            answer = timer(correct_bnmeani_bp)(dbndiff)
            max_time_author = max(max_time_author, timer.get())

            res = cmp(student, answer)
            if res == -1:
                self.assertIsInstance(student, torch.Tensor, msg=f'{name}: wrong type returned')
            maxdiff = max(maxdiff, res[2])
        self.assertLess(maxdiff, atol, msg=f"{name}: your maxdiff is more then {atol}, {maxdiff}")
        print_time(name, max_time_author, 'author')
        print_time(name, max_time_student, 'student')
