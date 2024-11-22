In this task you should implement `WordWindowClassifier`. All torch models should be inherited from `torch.Module` and implement `__init__` and `forward`.

As input, we take tensor of size `[batch_size x length_padded]` (in the previous task we padded sequence with `PADDING_TOKEN` to map each token to a window where this token is put in the center of this window).
After, as we discussed in the previous task, we unfold these windows from each sequence and for each window predict the probability of its center to be a location.
