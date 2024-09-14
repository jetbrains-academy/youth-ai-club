import unittest

from task import MultiHeadAttention
import torch
import torch.nn as nn
from solution import batch_size, block_size, max_iters, eval_interval, learning_rate, device, eval_iters, n_embd, \
     n_layer, dropout


class TestCase(unittest.TestCase):
    def test_init(self):
        multi_head = MultiHeadAttention(4, 64)
        self.assertIsInstance(multi_head.heads, nn.ModuleList, msg="MultiHeadAttention.heads is not a ModuleList")
        self.assertIsInstance(multi_head.proj, nn.Linear, msg="MultiHeadAttention.proj is not a Linear")
        self.assertIsInstance(multi_head.dropout, nn.Dropout, msg="MultiHeadAttention.dropout is not a Dropout")

        self.assertEqual(len(multi_head.heads), 4, msg="number of heads is not correct")
        self.assertEqual(multi_head.proj.in_features, 64, msg="number of input features is not correct")
        self.assertEqual(multi_head.proj.out_features, 64, msg="number of output features is not correct")
        self.assertEqual(multi_head.dropout.p, 0.0, msg="dropout probability")

    def test_forward(self):
        multi_head = MultiHeadAttention(4, 16)
        x = torch.randn(batch_size, block_size, n_embd)
        out = multi_head(x)  # student's out
        self.assertEqual(out.shape, torch.Size([batch_size, block_size, n_embd]), msg="output shape is not correct")
        correct_out = torch.cat([h(x) for h in multi_head.heads], dim=-1)
        correct_out = multi_head.dropout(multi_head.proj(correct_out))
        self.assertTrue(torch.allclose(out, correct_out, atol=1e-6), msg="output is not correct")
