import unittest
import torch

from task import preprocess_sentence, sentence2ids, UNKNOWN_TOKEN, LocationsDataset, collate_fn_sentence, WINDOW_SIZE


class TestCase(unittest.TestCase):
    def test_preprocess_sentence(self):
        sentence = "This is a sentence of several WORDS."
        gt_words = ["this", "is", "a", "sentence", "of", "several", "words."]
        predicted_words = preprocess_sentence(sentence)
        self.assertEqual(gt_words, predicted_words, msg="Wrong preprocess_sentence output")

    def test_sentence2ids(self):
        sentence = "This is a sentence of several WORDS."
        word_to_idx = {
            'this': 2,
            'sentence': 3,
            'a': 4,
            'is': 5,
            'several': 6,
        }
        gt_ids = [2, 5, 4, 3, 0, 6, 0]
        predicted_ids = sentence2ids(sentence, word_to_idx)
        self.assertEqual(predicted_ids, gt_ids, msg="Wrong sentence2ids output")

    def test_location_dataset(self):
        corpus = ["A B", "C D C"]
        locations = {"b", "c"}
        dataset = LocationsDataset(corpus, locations)
        self.assertEqual(corpus, dataset.corpus_)
        self.assertEqual(locations, dataset.locations_)

        gt_idx_to_word = ["<pad>", "<unk>", "a", "b", "c", "d"]
        self.assertEqual(gt_idx_to_word, sorted(dataset.idx_to_word_), msg="Wrong `words_set` in `LocationsDataset`")

        self.assertEqual(len(corpus), len(dataset))

        data = [
            (["a", "b"], [0, 1]),
            (["c", "d", "c"], [1, 0, 1]),
        ]
        for idx, (words, labels) in enumerate(data):
            word_ids = [dataset.word_to_idx_.get(word, 0) for word in words]
            self.assertEqual(word_ids, dataset[idx][0],
                             msg=f"Wrong ids for index {idx}. Corpus={corpus}, locations={locations}")
            self.assertEqual(labels, dataset[idx][1],
                             msg=f"Wrong labels for index {idx}. Corpus={corpus}, locations={locations}")

    def test_collate_fn_sentence(self):
        corpus = ["A B", "C D C"]
        locations = {"b", "c"}
        dataset = LocationsDataset(corpus, locations)

        word_idx_padded, labels, lengths = collate_fn_sentence([dataset[0], dataset[1]])

        self.assertIsInstance(word_idx_padded, torch.Tensor, msg="`word_idx_padded` should be Tensor")
        self.assertIsInstance(labels, torch.Tensor, msg="`labels` should be Tensor")
        self.assertIsInstance(lengths, torch.LongTensor, msg="`lengths` should be LongTensor")

        self.assertEqual(word_idx_padded.size(), torch.Size([2, 3 + 2 * WINDOW_SIZE])
                         , msg="Wrong size of `word_idx_padded`")
        self.assertEqual(labels.size(), torch.Size([2, 3]), msg="Wrong size of `labels`")
        self.assertEqual(lengths.size(), torch.Size([2]), msg="Wrong size of `lengths`")

        window = [1] * WINDOW_SIZE
        gt_words_ids0 = window + dataset[0][0] + [1] + window
        gt_words_ids1 = window + dataset[1][0] + window
        self.assertTrue(torch.allclose(word_idx_padded, torch.tensor([gt_words_ids0, gt_words_ids1])),
                        msg="`word_idx_padded` is incorrect")

        labels_ids0 = dataset[0][1] + [0]
        labels_ids1 = dataset[1][1]
        self.assertTrue(torch.allclose(labels, torch.tensor([labels_ids0, labels_ids1])),
                        msg="`labels` is incorrect")

        self.assertTrue(torch.allclose(lengths, torch.LongTensor([2, 3])), msg="`lengths` is incorrect")
