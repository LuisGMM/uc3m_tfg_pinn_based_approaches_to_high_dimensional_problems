import numpy as np
import torch
from torch.optim.lr_scheduler import LinearLR
from tfg import device

from tfg.nn.models import FFNN


def f(x: np.ndarray) -> np.ndarray:
    r, s, t, u, v, w = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

    return r * s + np.cos(t) + np.sin(2 * u) + v**3 + np.exp(w)


def main():
    def scheduler(opt):
        return LinearLR(
            opt,
            start_factor=1,
            end_factor=0.2,
            total_iters=5000 * 100,
            verbose=True,
        )

    np_x_train = np.random.uniform(-2, 2, (1000, 6))
    ffnn = FFNN.run(
        input_size=np_x_train.shape[1],
        hidden_size=64,
        output_size=1,
        learning_rate=0.001,
        num_epochs=5000,
        f=f,
        np_x_train=np_x_train,
        # batch_size=100,
        # shuffle=True,
        # scheduler=scheduler,
    )

    # TODO: Test it with some data
    np_x_test = np.random.uniform(-2, 2, (1000, 6))
    np_y_test = f(np_x_test)
    print(ffnn(torch.tensor(np_x_test, dtype=torch.float32).to(device)).detach().numpy() ** 2 - np_y_test**2)


if __name__ == '__main__':
    main()
