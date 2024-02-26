import numpy as np
import torch
from torch.optim.lr_scheduler import LinearLR
from tfg import device

from tfg.nn.models import FFNN


def f(x: np.ndarray) -> np.ndarray:
    r, s, t, u, v, w = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

    return r * s + np.cos(t) + np.sin(2 * u) + v**3 + np.exp(w)


def main() -> FFNN:
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
        batch_size=100,
        # shuffle=True,
        # scheduler=scheduler,
    )

    return ffnn


def test_main():
    ffnn = main()

    np_x_test = np.random.uniform(-2, 2, (1000, 6))
    np_y_test = f(np_x_test)
    output = ffnn(torch.tensor(np_x_test, dtype=torch.float32).to(device))
    loss = torch.nn.MSELoss()(output.flatten(), torch.tensor(np_y_test.flatten(), dtype=torch.float32).to(device))
    print(f'Loss: {loss.item()}')


def plot_over_r6():
    import matplotlib.pyplot as plt

    np_x_test = np.linspace(-2, 2, 1000)
    np_x_test = np.array([np_x_test] * 6).T
    np_y_test = f(np_x_test)

    ffnn = main()
    output = ffnn(torch.tensor(np_x_test, dtype=torch.float32).to(device))

    plt.title('FFNN vs True Function')
    plt.grid(axis='both', alpha=0.3)
    # Remove borders
    plt.gca().spines['top'].set_alpha(0.0)
    plt.gca().spines['bottom'].set_alpha(0.3)
    plt.gca().spines['right'].set_alpha(0.0)
    plt.gca().spines['left'].set_alpha(0.3)

    plt.xlabel('Line domain in 6D space (r)')
    plt.ylabel('FFNN(r) and f(r) values')

    plt.plot(np_x_test[:, 0], output.cpu().detach().numpy(), label='Learned Function', color='r')
    plt.plot(np_x_test[:, 0], np_y_test, label='True Function', linestyle='--', color='b')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_over_r6()
