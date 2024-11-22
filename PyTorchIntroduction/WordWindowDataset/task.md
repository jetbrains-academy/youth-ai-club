In this task, you should implement a typical torch data pipeline.
It consists of several parts:
1) `torch.utils.data.Dataset` - ancestor class for all your datasets.
Your datasets should be inherited from it and implement next methods
   * `__init__` - init method for your dataset
   * `__len__` - the length of your dataset (`len(dataset)`)
   * `__getitem__(idx)` - returns the element of your dataset with index `idx` (`dataset[idx]`)
2) `torch.utils.DataLoader` - base class for constructing batches from your dataset
   * `dataset` - your dataset inherited from `torch.utils.data.Dataset`
   * `batch_size` - size of each batch
   * `shuffle` - is data should be shuffled during each iteration
   * `num_workers` - number of workers to call `__getitem__` to construct batch
   * `collate_fn` - function that takes list of samples obtained from `__getitem__`
and constructs a batch consisting `torch.Tensor` items.
By default, take [this](https://pytorch.org/docs/stable/data.html#torch.utils.data.default_collate).
   * `pin_memory` - speeds up when using with GPU
   * `drop_last` - drops last batch.
It can be useful if you want all batches to be the same length — the last batch can be shorter than the previous.

We have a corpus of sentences for our task.
Each sentence we split into tokens — in our approach, we will lower all letters and split them into words.
For each word, we want to predict if it is a location or not.
Also, for each token we map a unique integer index.

Our dataset `__getitem__` method should return a list of token indices and a list of labels for each token (1 if location, 0 otherwise).

The problem is all token sequences have different length and can't be wrapped into `torch.Tensor`.
That's why we first pad each token sequence to the common length (maximum from all sequences) with `torch.nn.utils.rnn.pad_sequence`.

As we said, we want our model to predict for each word if it is a location or not.
To do this, it takes words around this token, so we should preprocess the data according to this idea.
In further task, we will cut windows of size `2 * WINDOW_SIZE + 1` with step $1$, 
and we want each token to be the center of some window.
That's why during preprocessing we should pad the token sequence with `PADDING_TOKEN` (in fact index of this token) - `WINDOW_SIZE` to the left and `WINDOW_SIZE` to the right.

All paddings can be done during your custom `collate_fn`.