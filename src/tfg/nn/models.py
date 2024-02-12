from typing import Callable
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


class FFNN(nn.Module):
    """
    A simple feedforward neural network with 2 layers (no hidden layers).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        training_size: int,
        *,
        activation_function: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()

        self.activation_function = activation_function
        self.training_size = training_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation_function(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.layers(x)

    def generate_data(self, f: Callable, np_x_train: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        np_f_train = f(np_x_train)

        x_train = torch.tensor(np_x_train, dtype=torch.float32).view(-1, self.input_size)
        f_train = torch.tensor(np_f_train, dtype=torch.float32).view(-1, self.output_size)

        return x_train, f_train

    def plot(self, f: Callable, np_x_test: np.ndarray) -> None:
        # Plotting the true function and the learned function
        x_test = torch.tensor(np_x_test, dtype=torch.float32).view(-1, self.input_size)
        with torch.no_grad():
            y_pred = self(x_test).numpy()

        plt.plot(x_test.numpy(), y_pred, label='Learned Function', color='r')
        plt.plot(x_test.numpy(), f(x_test.numpy()), label='True Function', linestyle='--', color='b')
        plt.legend()
        plt.show()

    def plot3d(self, f: Callable, np_x_test: np.ndarray, np_y_test: np.ndarray) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(np_x_test, np_y_test)
        Z = f(np.array([X.ravel(), Y.ravel()]).T).reshape(X.shape)
        ax.plot_surface(X, Y, Z, alpha=0.5, cmap=plt.get_cmap('viridis'))

        Z = (
            self(torch.tensor(np.array([X.ravel(), Y.ravel()]).T, dtype=torch.float32))
            .detach()
            .numpy()
            .reshape(X.shape)
        )
        ax.plot_surface(X, Y, Z, alpha=0.5, cmap=plt.get_cmap('copper'))

        plt.show()

    @staticmethod
    def run(
        input_size: int,
        hidden_size: int,
        output_size: int,
        training_size: int,
        learning_rate: float,
        num_epochs: int,
        f: Callable,
        np_x_train: np.ndarray,
        activation_function: type[nn.Module] = nn.Tanh,
        criterion: type[nn.Module] = nn.MSELoss,
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        scheduler: Callable[
            [torch.optim.Optimizer],
            torch.optim.lr_scheduler.LRScheduler,
        ]
        | None = None,
        log_interval: int = 100,
    ):
        model = FFNN(input_size, hidden_size, output_size, training_size, activation_function=activation_function)
        _criterion = criterion()
        _optimizer = optimizer(model.parameters(), lr=learning_rate)

        if scheduler is not None:
            _scheduler = scheduler(_optimizer)
        else:
            _scheduler = None

        x_train, f_train = model.generate_data(f, np_x_train)

        for epoch in range(num_epochs):
            outputs = model(x_train)
            loss = _criterion(outputs, f_train)

            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()

            if _scheduler is not None:
                _scheduler.step()

            log_str = f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}'

            if _scheduler is not None:
                log_str += f', LR: {_scheduler.get_last_lr()}'

            if (epoch + 1) % log_interval == 0:
                print(log_str)

        return model
