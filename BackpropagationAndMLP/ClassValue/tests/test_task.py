import unittest
from task import Value


class TestCase(unittest.TestCase):

    a = Value(2.0)
    b = Value(-3.0)

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
