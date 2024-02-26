from typing import Callable
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from tfg import device


class FFNN(nn.Module):
    """
    A simple feedforward neural network with 2 layers (no hidden layers).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        activation_function: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()

        self.activation_function = activation_function
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
        x_test = torch.tensor(np_x_test, dtype=torch.float32).view(-1, self.input_size).to(device)
        with torch.no_grad():
            y_pred = self(x_test).cpu().numpy()

        x_test = x_test.cpu().numpy()

        plt.plot(x_test, y_pred, label='Learned Function', color='r')
        plt.plot(x_test, f(x_test), label='True Function', linestyle='--', color='b')
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
            .cpu()
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
        batch_size: int | None = None,
        shuffle: bool = False,
        log_interval: int = 1,
    ):
        model = FFNN(
            input_size,
            hidden_size,
            output_size,
            activation_function=activation_function,
        ).to(device)
        _criterion = criterion()
        _optimizer = optimizer(model.parameters(), lr=learning_rate)  # type: ignore

        if scheduler is not None:
            _scheduler = scheduler(_optimizer)
        else:
            _scheduler = None

        x_train, f_train = model.generate_data(f, np_x_train)

        batch_size = batch_size if batch_size is not None else len(x_train)
        num_batches = len(x_train) // batch_size

        for epoch in range(num_epochs):
            if shuffle:
                indices = torch.randperm(len(x_train))
                x_train_modified = x_train[indices]
                f_train_modified = f_train[indices]

            else:
                x_train_modified = x_train
                f_train_modified = f_train

            for batch_i in range(num_batches):
                start_idx = batch_i * batch_size
                end_idx = (batch_i + 1) * batch_size
                x_batch = x_train_modified[start_idx:end_idx].to(device)
                f_batch = f_train_modified[start_idx:end_idx].to(device)

                outputs = model(x_batch)
                loss = _criterion(outputs, f_batch)

                _optimizer.zero_grad()
                loss.backward()
                _optimizer.step()

                if _scheduler is not None:
                    _scheduler.step()

                log_str = f'Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item():.4f} Batch: [{batch_i+1:>5d}/{num_batches:>5d}]'

                if _scheduler is not None:
                    log_str += f' LR: {_scheduler.get_last_lr()}'

                if (epoch + 1) % log_interval == 0:
                    print(log_str)

        return model
