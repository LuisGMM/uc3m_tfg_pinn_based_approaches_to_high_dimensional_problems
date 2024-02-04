import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


n = 2
TRAINING_SIZE = 1000
INPUT_SIZE = 1
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 5000


def f(x: np.ndarray, n: float = n) -> np.ndarray:
    return np.cos(n * x)


class Model(nn.Module):
    """
    A simple feedforward neural network with 2 layers (no hidden layers).
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            # nn.ReLU(), # NOTE: Incredible difference
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.layers(x)


def main(plot: bool = False):
    np_x_train = np.random.uniform(-2 * np.pi, 2 * np.pi, TRAINING_SIZE)
    np_y_train = f(np_x_train)

    x_train = torch.tensor(np_x_train, dtype=torch.float32).view(-1, 1)
    y_train = torch.tensor(np_y_train, dtype=torch.float32).view(-1, 1)

    model = Model(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        outputs = model(x_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

    if plot:
        # Plotting the true function and the learned function
        x_test = torch.linspace(-2 * np.pi, 2 * np.pi, 1000).view(-1, 1)
        with torch.no_grad():
            y_pred = model(x_test).numpy()

        plt.plot(x_test.numpy(), y_pred, label='Learned Function', color='r')
        plt.plot(x_test.numpy(), f(x_test.numpy()), label='True Function', linestyle='--', color='b')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main(plot=True)
