import torch.nn as nn


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
