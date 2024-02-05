import numpy as np

from tfg.nn.models import FFNN


def f(x: np.ndarray) -> np.ndarray:
    return x[:, 0] ** 2 + x[:, 1] ** 2


def main(plot: bool = False):
    ffnn = FFNN.run(
        input_size=2,
        hidden_size=64,
        output_size=1,
        training_size=1000,
        learning_rate=0.001,
        num_epochs=5000,
        f=f,
        np_x_train=np.random.uniform(-2, 2, (1000, 2)),
    )

    np_x_test = np.linspace(-2, 2, 100)
    np_y_test = np.linspace(-2, 2, 100)

    if plot:
        ffnn.plot3d(f, np_x_test, np_y_test)


if __name__ == '__main__':
    main(plot=True)
