In the next tasks you will work with the PyTorch library.
It is a very powerful tool for working with machine learning models in Python.

In this lesson you are invited to run different code fragments in **Jupiter notebook** to master the basic capabilities of this library.

### Tensors

Tensors are a specialized data structure that are very similar to arrays and matrices.
In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.

Tensors can be **initialized** in various ways. Take a look at the following examples(1-3):

Tensor **attributes** describe their shape, datatype, and the device on which they are stored(4).

Tensors contain over 100 **operations**, including arithmetic, linear algebra, matrix manipulation.
You can see some basic operations in examples (5-8)

Tensors on the CPU and **NumPy arrays** can share their underlying memory locations, and changing one will change the other(9-11).

### Get Device for Training

Neural networks consist of layers/modules that perform operations on data.
The `torch.nn` namespace provides all the building blocks you need to build your own neural network. 
Every module in PyTorch subclasses the `nn.Module`. A neural network is a module itself that consists of other modules (layers).
This nested structure allows for building and managing complex architectures easily.

We want to be able to train our model on a hardware accelerator like the GPU or MPS, if available.
Let’s check to see if `torch.cuda` or `torch.backends.mps` are available, otherwise we use the CPU (12).

### Define the Class

We define our neural network by subclassing `nn.Module`, and initialize the neural network layers in `__init__`.
Every `nn.Module` subclass implements the operations on input data in the forward method.

We create an instance of NeuralNetwork, and move it to the device, and print its structure (13).

To use the model, we pass it the input data. This executes the model’s forward, along with some background operations. Do not call `model.forward()` directly!

Calling the model on the input returns a 2-dimensional tensor with `dim=0` corresponding to each output of 10 raw predicted values 
for each class, and `dim=1` corresponding to the individual values of each output.
We get the prediction probabilities by passing it through an instance of the `nn.Softmax` module (14).

### Model Layers

Let’s break down the layers in the **FashionMNIST** model.
To illustrate it, we will take a sample minibatch of 3 images of size 28x28 and see what happens to it as we pass it through the network (15).

We initialize the `nn.Flatten` layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values (the minibatch dimension (at dim=0) is maintained) (16).

The **linear layer** is a module that applies a linear transformation on the input using its stored weights and biases (17).

Non-linear activations are what create the complex mappings between the model’s inputs and outputs. They are applied after linear transformations to introduce nonlinearity, helping neural networks learn a wide variety of phenomena.

In this model, we use `nn.ReLU` between our linear layers, but there’s other activations to introduce non-linearity in your model (18).

`nn.Sequential` is an ordered container of modules. The data is passed through all the modules in the same order as defined. You can use sequential containers to put together a quick network like seq_modules (19).

The last linear layer of the neural network returns logits - raw values in `[-infty, infty]` - which are passed to the `nn.Softmax` module.
The logits are scaled to values `[0, 1]` representing the model’s predicted probabilities for each class. `dim` parameter indicates the dimension along which the values must sum to 1 (20).

### Model Parameters

Many layers inside a neural network are parameterized, i.e. have associated weights and biases that are optimized during training.
Subclassing `nn.Module` automatically tracks all fields defined inside your model object, and makes all parameters accessible using your model’s `parameters()` or `named_parameters()` methods (21).

In this example, we iterate over each parameter, and print its size and a preview of its values.