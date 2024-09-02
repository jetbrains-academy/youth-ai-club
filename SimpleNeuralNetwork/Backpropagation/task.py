import numpy as np

from SimpleNeuralNetwork.Forward.task import Forward

def sigmoid_derivative(z):
    return z * (1 - z)


class Backpropagation(Forward):
    def train(self, inputs, targets, epochs, lr):
        predicted_outputs = None
        for epoch_num in range(epochs):
            predicted_outputs = self.forward(inputs)

            # Loss
            loss = 0.5 * (targets - predicted_outputs) ** 2
            mse_loss = loss.sum() / len(inputs)
            if epoch_num % 1000 == 0:
                print(f"Epoch {epoch_num}: mse_loss = {mse_loss}")

            loss_by_outputs = -(targets - predicted_outputs)
            predicted_outputs_derivative = sigmoid_derivative(predicted_outputs)

            loss_by_output_bias = predicted_outputs_derivative * loss_by_outputs

            #  output_weights(2, 1) = hidden_outputs(4, 2).transpose @ (4, 1)
            loss_by_output_weights = self.hidden_outputs.transpose().dot(
                predicted_outputs_derivative * loss_by_outputs
            )

            #  hidden_outputs(4, 2) = (4, 1) @ output_weights(2, 1).transpose
            loss_by_hidden_outputs = (predicted_outputs_derivative * loss_by_outputs).dot(
                self.output_weights.transpose()
            )

            hidden_outputs_derivative = sigmoid_derivative(self.hidden_outputs)
            loss_by_hidden_bias = hidden_outputs_derivative * loss_by_hidden_outputs

            # hidden_weights(2, 2) = inputs(4, 2).transpose @ (4, 2)
            loss_by_hidden_weights = inputs.transpose().dot(
                hidden_outputs_derivative * loss_by_hidden_outputs
            )

            # Updating weights and biases
            self.output_bias -= lr * loss_by_output_bias.sum(axis=0)
            self.output_weights -= lr * loss_by_output_weights
            self.hidden_bias -= lr * loss_by_hidden_bias.sum(axis=0)
            self.hidden_weights -= lr * loss_by_hidden_weights
        return predicted_outputs


if __name__ == '__main__':
    # Input datasets
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])

    # Train parameters
    epochs = 10000
    lr = 0.1

    model = Backpropagation(2, 2, 1)
    predicted_outputs = model.train(inputs, targets, epochs, lr)

    print('')
    print("Final hidden weights: ", end='')
    print(*model.hidden_weights)
    print("Final hidden bias: ", end='')
    print(*model.hidden_bias)
    print("Final output weights: ", end='')
    print(*model.output_weights)
    print("Final output bias: ", end='')
    print(*model.output_bias)

    print(f"\nOutput from neural network after {epochs} epochs: ")
    print(predicted_outputs)
