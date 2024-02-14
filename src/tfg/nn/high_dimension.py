import numpy as np

from tfg.nn.models import FFNN


def f(x: np.ndarray) -> np.ndarray:
    r, s, t, u, v, w = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

    return r * s + np.cos(t) + np.sin(2 * u) + v**3 + np.exp(w)


def main():
    np_x_train = np.random.uniform(-2, 2, (1000, 6))
    FFNN.run(
        input_size=np_x_train.shape[1],
        hidden_size=64,
        output_size=1,
        learning_rate=0.001,
        num_epochs=5000,
        f=f,
        np_x_train=np_x_train,
    )


if __name__ == '__main__':
    main()
