import unittest

from task import Block, FeedFoward
from solution import block_size, n_embd, dropout, batch_size
from torch.nn import functional as F
from Transformers.MultiHead.task import MultiHeadAttention
import torch
import torch.nn as nn


class TestCase(unittest.TestCase):
    def test_block(self):
        n_head = 4
        n_embd = 64
        block = Block(n_embd, n_head)
        self.assertIsInstance(block.sa, MultiHeadAttention, msg="Block.sa is not a MultiHeadAttention")
        self.assertIsInstance(block.ffwd, FeedFoward, msg="Block.ffwd is not a FeedFoward")
        self.assertIsInstance(block.ln1, nn.LayerNorm, msg="Block.ln1 is not a LayerNorm")
        self.assertIsInstance(block.ln2, nn.LayerNorm, msg="Block.ln2 is not a LayerNorm")
        x = torch.randn(batch_size, block_size, n_embd)
        y = block(x)  # student's solution
        ans = x + block.sa(block.ln1(x))
        ans = ans + block.ffwd(block.ln2(ans))
        self.assertTrue(torch.allclose(y, ans), msg="Block.forward is incorrect")