import unittest
import numpy as np

from task import MLP, Layer, BatchNorm


class TestBatchNorm(unittest.TestCase):

    def test_parameters(self):
        bn = BatchNorm(dim=3)
        params = bn.parameters()
        self.assertEqual(len(params), 2)
        self.assertTrue(all(isinstance(param, np.ndarray) for param in params))

    def test_training_mode(self):
        batch_norm = BatchNorm(dim=3)
        batch_norm.training = True
        self.assertTrue(batch_norm.training)
        x = np.random.randn(3, 3, 3)
        output = batch_norm(x)
        self.assertEqual(output.shape, (3, 3, 3))

    def test_inference_mode(self):
        batch_norm = BatchNorm(dim=3)
        batch_norm.training = False
        self.assertFalse(batch_norm.training)
        x = np.random.randn(3, 3, 3)
        output = batch_norm(x)
        self.assertEqual(output.shape, (3, 3, 3))


class TestMLP(unittest.TestCase):

    def test_parameters(self):
        mlp = MLP(nin=2, nouts=[3, 2])
        params = mlp.parameters()
        self.assertEqual(len(params), 14)

    def test_init(self):
        mlp = MLP(nin=3, nouts=[2, 3])
        self.assertEqual(len(mlp.layers), 3)

    def test_instance(self):
        mlp = MLP(nin=3, nouts=[2, 3])
        self.assertIsInstance(mlp.layers[0], Layer)
        self.assertIsInstance(mlp.layers[1], BatchNorm)
        self.assertIsInstance(mlp.layers[2], Layer)


if __name__ == '__main__':
    unittest.main()
