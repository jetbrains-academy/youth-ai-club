import unittest
import torch
import torch.nn as nn

from task import WordWindowClassifier


# todo: replace this with an actual test
class TestCase(unittest.TestCase):
    def test_word_window_classifier_init(self):
        model = WordWindowClassifier(3, 16, 32, False, 10, 0)

        self.assertIsInstance(model.embeds, nn.Embedding, msg="WordWindowClassifier.embeds should be nn.Embedding")
        self.assertEqual(model.embeds.num_embeddings, 10,
                         msg="Wrong WordWindowClassifier.embeds.num_embeddings")
        self.assertEqual(model.embeds.embedding_dim, 16,
                         msg="Wrong WordWindowClassifier.embeds.embedding_dim")
        self.assertEqual(model.embeds.padding_idx, 0,
                         msg="Wrong WordWindowClassifier.embeds.padding_idx")

        self.assertIsInstance(model.hidden_layer, nn.Sequential,
                              msg="WordWindowClassifier.hidden_layer should be nn.Sequential")
        self.assertIsInstance(model.hidden_layer[0], nn.Linear,
                              msg="WordWindowClassifier.hidden_layer[0] should be nn.Linear")
        self.assertEqual(model.hidden_layer[0].in_features, (2 * 3 + 1) * 16,
                              msg="Incorrect WordWindowClassifier.hidden_layer[0].in_features")
        self.assertIsInstance(model.hidden_layer[1], nn.Tanh,
                              msg="WordWindowClassifier.hidden_layer[1] should be nn.Tanh")

        self.assertIsInstance(model.output_layer, nn.Sequential,
                              msg="WordWindowClassifier.output_layer should be nn.Sequential")
        self.assertIsInstance(model.output_layer[0], nn.Linear,
                              msg="WordWindowClassifier.output_layer[0] should be nn.Linear")
        self.assertEqual(model.output_layer[0].in_features, 32,
                              msg="Incorrect WordWindowClassifier.output_layer[0].in_features")
        self.assertEqual(model.output_layer[0].out_features, 1,
                              msg="Incorrect WordWindowClassifier.output_layer[0].out_features")
        self.assertIsInstance(model.output_layer[1], nn.Sigmoid,
                              msg="WordWindowClassifier.output_layer[1] should be nn.Sigmoid")

    def test_word_window_classifier_forward(self):
        window_size = 3
        model = WordWindowClassifier(window_size, 16, 32, False, 10, 0)

        inputs = torch.randint(low=0, high=9, size=(8, 10))
        predicted_output = model(inputs)

        token_windows = inputs.unfold(1, 2 * window_size + 1, 1)
        embedded_windows = model.embeds(token_windows)
        embedded_windows = embedded_windows.view(embedded_windows.size(0), embedded_windows.size(1), -1)
        layer_1 = model.hidden_layer(embedded_windows)
        output = model.output_layer(layer_1)
        output = output.view(output.size(0), output.size(1))

        self.assertEqual(output.size(), torch.Size([8, 10 - (2 * window_size + 1) + 1]),
                         msg="Wrong size of the output")
        self.assertTrue(torch.allclose(predicted_output, output),
                        msg="Output of WordWindowClassifier.forward is incorrect")
