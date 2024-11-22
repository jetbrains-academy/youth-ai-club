In this task you should implement simple `MultilayerPerceptron` model.
It should apply consistently next operations
1. `torch.nn.Linear`
2. `torch.nn.ReLU`
3. `torch.nn.Linear`
4. `torch.nn.Sigmoid`

`__init__` method will obtain `input_size` - the embedding dimension of input and `hidden_size` - the hidden embedding dimension after first `Linear`.

The input and output tensors should have the same shape (the model preserves embedding dimension).
