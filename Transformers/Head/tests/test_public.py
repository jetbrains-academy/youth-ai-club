import unittest

from task import Head
from solution import batch_size, block_size, max_iters, eval_interval, learning_rate, device, eval_iters, n_embd, \
    n_head, n_layer, dropout
from torch.nn import functional as F
import torch


class TestCase(unittest.TestCase):
    def test_init(self):
        head = Head(10)
        self.assertIsInstance(head.key, torch.nn.Linear, "Head.key is not a Linear layer")
        self.assertIsInstance(head.query, torch.nn.Linear, "Head.query is not a Linear layer")
        self.assertIsInstance(head.value, torch.nn.Linear, "Head.value is not a Linear layer")
        self.assertIsInstance(head.tril, torch.Tensor, "Head.tril is not a Tensor")
        self.assertIsInstance(head.dropout, torch.nn.Dropout, "Head.dropout is not a Dropout layer")
        self.assertEqual(head.tril.shape, (block_size, block_size), "Head.tril has wrong shape")
        self.assertEqual(head.key.bias, None, "Head.key.bias is not None")
        self.assertEqual(head.query.bias, None, "Head.query.bias is not None")
        self.assertEqual(head.value.bias, None, "Head.value.bias is not None")
        self.assertEqual(head.key.weight.shape, (10, n_embd), "Head.key.weight has wrong shape")
        self.assertEqual(head.query.weight.shape, (10, n_embd), "Head.query.weight has wrong shape")
        self.assertEqual(head.value.weight.shape, (10, n_embd), "Head.value.weight has wrong shape")
        self.assertEqual(head.dropout.p, dropout, "Head.dropout.p has wrong value")
    def test_forward(self):
        head = Head(10)
        x = torch.rand(batch_size, block_size, n_embd)
        out = head(x) # student's out
        B, T, C = x.shape
        k = head.key(x)
        q = head.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(head.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = head.dropout(wei)
        v = head.value(x)
        correct_out = wei @ v
        self.assertEqual(out.shape, correct_out.shape, "Head.forward has wrong shape")
        self.assertTrue(torch.allclose(out, correct_out), "Head.forward has wrong values")
