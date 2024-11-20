import unittest

from torch.utils.data import DataLoader

from PyTorchIntroduction.WordWindowTraining.task import train
from PyTorchIntroduction.WordWindowDataset.task import corpus, locations, LocationsDataset, collate_fn_sentence, WINDOW_SIZE
from PyTorchIntroduction.WordWindowClassifier.task import WordWindowClassifier


# todo: replace this with an actual test
class TestCase(unittest.TestCase):
    def test_training(self):
        dataset = LocationsDataset(corpus, locations)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn_sentence)

        model = WordWindowClassifier(WINDOW_SIZE, 16, 8,
                                     False, len(dataset.idx_to_word_), pad_idx=1)

        train_losses, val_losses, val_accs = train(model, loader, loader, 15, validate_every_n_epochs=5)

        self.assertEqual(val_accs[-1], 1., msg=f"On toy data where test_loader=train_loader last val acc shiuld be 1., got {val_accs[-1]}")
