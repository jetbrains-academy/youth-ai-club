import unittest
from task import Value


class TestCase(unittest.TestCase):
    a = Value(2.0)
    b = Value(-3.0)

    x1 = Value(2.0)
    x1.grad = 0.0
    x2 = Value(0.0)
    x2.grad = 0.0
    w1 = Value(-3.0)
    w1.grad = 0.0
    w2 = Value(1.0)
    w2.grad = 0.0
    c = Value(6.8813735870195432)
    c.grad = 0.0

    x1w1 = x1 * w1
    x1w1.grad = 0.0
    x2w2 = x2 * w2
    x2w2.grad = 0.0
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.grad = 0.0
    n = x1w1x2w2 + c
    n.grad = 0.0
    e = (2 * n).exp()
    e.grad = 0.0
    o = (e - 1) / (e + 1)
    o.grad = 0.0

    o.backward()

    def test_add(self):
        self.assertEqual((self.a + self.b).data, -1.0, msg="add operation is incorrect")

    def test_mul(self):
        self.assertEqual((self.a * self.b).data, -6.0, msg="multiplication operation is incorrect")

    def test_sub(self):
        self.assertEqual((self.a - self.b).data, 5.0, msg="subtract operation is incorrect")

    def test_div(self):
        self.assertEqual((self.b / self.a).data, -1.5, msg="division operation is incorrect")

    def test_tanh(self):
        self.assertEqual((self.a.tanh()).data, 0.9640275800758169, msg="tanh operation is incorrect")

    def test_exp(self):
        self.assertEqual(self.a.exp().data, 7.38905609893065, msg="exp operation is incorrect")

    def test_add_number(self):
        self.assertEqual((self.a + 2.0).data, 4.0, msg="add operation is incorrect")

    def test_mul_number(self):
        self.assertEqual((self.a * 3.0).data, 6.0, msg="multiplication operation is incorrect")

    def test_sub_number(self):
        self.assertEqual((self.a - 1.0).data, 1.0, msg="subtract operation is incorrect")

    def test_div_number(self):
        self.assertEqual((self.b / 2.0).data, -1.5, msg="division operation is incorrect")

    def test_add_number_reverse(self):
        self.assertEqual((3.0 + self.b).data, 0.0, msg="add operation is incorrect")

    def test_mul_number_reverse(self):
        self.assertEqual((2.0 * self.b).data, -6.0, msg="multiplication operation is incorrect")

    def test_sub_number_reverse(self):
        self.assertEqual((2.0 - self.b).data, 5.0, msg="subtract operation is incorrect")

    def test_div_number_reverse(self):
        self.assertEqual((3.0 / self.b).data, -1.0, msg="division operation is incorrect")

    def test_grad_1(self):
        self.assertEqual(self.x2.grad, 0.5, msg="first gradient is incorrect")

    def test_grad_2(self):
        self.assertEqual(self.w2.grad, 0.0, msg="second gradient is incorrect")

    def test_grad_3(self):
        self.assertEqual(self.x1.grad, -1.5, msg="third gradient is incorrect")

    def test_grad_4(self):
        self.assertEqual(self.w1.grad, 1.0, msg="fourth gradient is incorrect")
