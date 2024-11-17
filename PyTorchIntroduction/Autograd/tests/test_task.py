import math
import unittest

from task import count_grad


# todo: replace this with an actual test
class TestCase(unittest.TestCase):
    def test_count_grad(self):
        inputs = [0., 1., -0.5, -1.]
        answers = [
            (1., 0.),
            (0., 2),
            (1 / math.sqrt(math.e), -1.),
            (1 / math.e, -2),
        ]
        for input_value, (ans1, ans2) in zip(inputs, answers):
            pred1, pred2 = count_grad(input_value)
            self.assertAlmostEqual(pred1, ans1, 3, msg=f"First output: {input_value=}")
            self.assertAlmostEqual(pred2, ans2, 3, msg=f"Second output: {input_value=}")
