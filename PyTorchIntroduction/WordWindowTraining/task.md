In this task we will finally train our model!

Here are some steps you should implement in typical torch pipelines:
1) `torch.optim.Optimizer` - you can choose one of many optimizers torch library provides. In this task we chose `SGD`.
It always takes `nn.Module().parmeters()` - the list of all tensors of your model weights.
And our optimizer will optimize these weights during training. Every training step you should
   1) Clean gradients of all weights (like we did in Autograd task `t_input.grad = None`).
`optimizer.zero_grad()` method runs over all weights and cleans their grads.
   2) Count loss between your prediction and ground truth label
   3) Run `loss.backward()` to calculate grads (like we did in Autograd task)
   4) Call `optimizer.step()` - it will run over all weights and optimize using calculated grads.
For SGD it will do `tensor = tensor - lr * grad` (but the formula depends on the parameters, usually it more complicated).
2) `torch.nn.BCELoss` - this class implements simple Binary Cross Entropy loss for classification task.
you can find many other losses provided by torch.

Every `validate_every_n_epochs` epochs we will validate our model on `test_loader`.
For this, we will do the same steps while training but without calculating gradients and optimizing weights.
That's why for speeding up the process we use decorator `@torch.no_grad()`.
Operations with tensors automatically calculating graph to further calculate the grads.
We don't need the grads $\Rightarrow$ we don't need the graph and this decorator turns off graph calculation inside function it decorates.
You cal also use this trick like this:

```python
with torch.no_grad():
    # your code that doesn't need grads for tensors
```

Another moment we want to highlight is using different modes for our model: `train` an `eval`.
We can switch between them by calling corresponding methods. In this task it is not essential but some layers behave different in `train` and `eval` mode and using this trick will improve metrics. 

`device` parameter can be used to train on GPU (doesn't need in this task).

To test your solution we will build loaders and model using last 2 tasks.
The common practice to check your model is to overfit on small amount of data and check tha metrics are as high as they can be in theory.
So, we take small corpus of sentence from previous task, build `train_loader` and use the same `test_loader`.
Your final validation accuracy should be `100%`.
Such test doesn't guarantee that you model works good, but it shows that you highly possible don't have bugs in your pipeline.

Here is our code or testing your solution, please, don't cheat :)

```python
from torch.utils.data import DataLoader

from PyTorchIntroduction.WordWindowTraining.task import train
from PyTorchIntroduction.WordWindowDataset.task import corpus, locations, LocationsDataset, collate_fn_sentence, WINDOW_SIZE
from PyTorchIntroduction.WordWindowClassifier.task import WordWindowClassifier

dataset = LocationsDataset(corpus, locations)
loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn_sentence)

model = WordWindowClassifier(WINDOW_SIZE, 16, 8,
                            False, len(dataset.idx_to_word_), pad_idx=1)

train_losses, val_losses, val_accs = train(model, loader, loader, 15, validate_every_n_epochs=5)

self.assertEqual(val_accs[-1], 1., msg=f"On toy data where test_loader=train_loader last val acc shiuld be 1., got {val_accs[-1]}")
```
