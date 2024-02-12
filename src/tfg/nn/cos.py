import numpy as np
import torch

from tfg.nn.models import FFNN
import torch.nn as nn


def f(x: np.ndarray, n: float) -> np.ndarray:
    return np.cos(n * x)


def main(plot: bool = False):
    n = 2
    TRAINING_SIZE = 1000
    INPUT_SIZE = 1
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 1
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5000

    def scheduler(opt):
        return torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=1,
            end_factor=0.2,
            total_iters=5000,
            verbose=True,
        )

    def f(x: np.ndarray) -> np.ndarray:
        return np.cos(n * x)

    ffnn = FFNN.run(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        training_size=TRAINING_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        activation_function=nn.ReLU,
        f=f,
        np_x_train=np.random.uniform(-2 * np.pi, 2 * np.pi, TRAINING_SIZE),
        scheduler=scheduler,
    )

    if plot:
        np_x_test = np.linspace(-2 * np.pi, 2 * np.pi, TRAINING_SIZE)
        ffnn.plot(f, np_x_test)


if __name__ == '__main__':
    main(plot=True)
