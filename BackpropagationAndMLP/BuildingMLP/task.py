from BackpropagationAndMLP.AutoBackpropagation.task import Value
import random
import numpy as np


class BatchNorm:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.gamma = np.array([Value(1.0) for _ in range(dim)])
        self.beta = np.array([Value(0.0) for _ in range(dim)])
        self.training = True
        self.momentum = momentum
        self.eps = eps
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)

    def __eq__(self, other):
        return isinstance(other, BatchNorm) and \
            np.array_equal(self.gamma, other.gamma) and \
            np.array_equal(self.beta, other.beta)

    def __call__(self, x):
        if self.training:
            assert x.ndim == 3
            axis = (0, 1)
            xmean = x.mean(axis)
            xvar = x.var(axis)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / np.sqrt(xvar + self.eps)  # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        return self.out

    def __str__(self):
        return f"BatchNorm(dim={len(self.gamma)})"

    def parameters(self):
        return [self.gamma, self.beta]


class Neuron:

    def __init__(self, nin):
        self.w = np.array([Value(random.uniform(-1, 1)) for _ in range(nin)])
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def __eq__(self, other):
        return isinstance(other, Neuron) and \
            np.array_equal(self.w, other.w) and \
            self.b == other.b

    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, nin, nout):
        self.neurons = np.array([Neuron(nin) for _ in range(nout)])

    def __eq__(self, other):
        return isinstance(other, Layer) and \
            np.array_equal(self.neurons, other.neurons)

    def __call__(self, x):
        outs = np.array([n(x) for n in self.neurons])
        return outs[0] if len(outs) == 1 else outs

    def __str__(self):
        return f"Layer({len(self.neurons[0].w)}, {len(self.neurons)})"

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts) - 1):  # 1 time
            self.layers.append(Layer(sz[i], sz[i + 1]))  # Layer (3, 2)
            self.layers.append(BatchNorm(sz[i + 1]))  # BatchNorm (2)
        self.layers.append(Layer(sz[-2], sz[-1]))  # Layer (2, 3)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
