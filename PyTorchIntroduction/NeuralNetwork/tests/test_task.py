import unittest
import torch
import torch.nn as nn

from task import MultilayerPerceptron


# todo: replace this with an actual test
class TestCase(unittest.TestCase):
    def test_multilayer_perceptron(self):
        net = MultilayerPerceptron(64, 16)
        self.assertIsInstance(net.model, nn.Sequential, msg="self.model is not nn.Sequential")

        gt_model = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.Sigmoid(),
        )

        for i, (student_module, gt_module) in enumerate(zip(net.model, gt_model)):
            self.assertIsInstance(student_module, gt_module.__class__, msg=f"Layer {i}: wrong layer class")
            if isinstance(student_module, nn.Linear):
                self.assertEqual(student_module.in_features, gt_module.in_features, msg=f"Layer {i}: wrong in_features")
                self.assertEqual(student_module.out_features, gt_module.out_features,
                                 msg=f"Layer {i}: wrong out_features")
                self.assertNotEqual(student_module.bias, None, msg=f"Layer {i}: wrong bias")

    def test_multilayer_perceptron_forward(self):
        net = MultilayerPerceptron(64, 16)

        x = torch.randn(8, 64)
        student_out = net(x)
        gt_out = net.model(x)

        self.assertTrue(torch.allclose(student_out, gt_out), msg="Wrong model output")
