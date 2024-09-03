import unittest
import torch

from task import tokenization, embeddings
from solution import correct_tokenization, correct_embeddings


class TestCase(unittest.TestCase):
    def test_tokenization1(self):
        words = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce vel ex vitae erat convallis! Maecenas?'
        self.assertEqual(tokenization(words), correct_tokenization(words), msg='Incorrect tokenization')

    def test_tokenization2(self):
        words = ''
        self.assertEqual(tokenization(words), correct_tokenization(words), msg='Incorrect tokenization')

    def test_tokenization3(self):
        words = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@$%^&*()-_=+[{]};:,<.>/?`'
        self.assertEqual(tokenization(words), correct_tokenization(words), msg='Incorrect tokenization')

    def test_embeddings1(self):
        vocab_size = 10
        n_embd = 5
        seed = 42
        self.assertTrue(
            torch.allclose(embeddings(vocab_size, n_embd, seed), correct_embeddings(vocab_size, n_embd, seed)),
            msg='Incorrect embeddings')

    def test_embeddings2(self):
        vocab_size = 1
        n_embd = 1
        seed = 49683
        self.assertTrue(
            torch.allclose(embeddings(vocab_size, n_embd, seed), correct_embeddings(vocab_size, n_embd, seed)),
            msg='Incorrect embeddings')

    def test_embeddings3(self):
        vocab_size = 100
        n_embd = 100
        seed = 3456
        self.assertTrue(
            torch.allclose(embeddings(vocab_size, n_embd, seed), correct_embeddings(vocab_size, n_embd, seed)),
            msg='Incorrect embeddings')