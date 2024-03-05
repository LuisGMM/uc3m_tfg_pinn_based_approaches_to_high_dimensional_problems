import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ODE(nn.Module):
    def __init__(self):
        super(ODE, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def solve_ode(np_x_train, np_f_train, u_a, u_b, num_epochs=1000, lr=0.01):
        model = ODE()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        x_train = torch.tensor(np_x_train, dtype=torch.float32, requires_grad=True).view(-1, 1)
        f_train = torch.tensor(np_f_train, dtype=torch.float32).view(-1, 1)

        for _ in range(num_epochs):
            optimizer.zero_grad()
            u_train = model(x_train)

            # Compute gradient of u_train with respect to x_train
            u_grad = torch.autograd.grad(
                u_train,
                x_train,
                grad_outputs=torch.ones_like(u_train),
                create_graph=True,
                retain_graph=True,
            )[0]

            # Predict values at boundary points
            u_a_pred = model(torch.tensor([[a]], dtype=torch.float32))
            u_b_pred = model(torch.tensor([[b]], dtype=torch.float32))

            # Compute loss using u_grad
            loss = (
                criterion(u_grad, f_train)
                + criterion(u_a_pred, torch.tensor([[u_a]], dtype=torch.float32))
                + criterion(u_b_pred, torch.tensor([[u_b]], dtype=torch.float32))
            )

            loss.backward()
            optimizer.step()

        return model


if __name__ == '__main__':
    # Define the interval [a, b] and the boundary conditions u(a) and u(b)
    a = 0
    b = 2.0

    def u_solution(x):
        return x**4

    u_a = u_solution(a)
    u_b = u_solution(b)

    def f(x):
        return 4 * x**3

    np_x_train = np.random.uniform(a, b, 1000)
    np_x_train.sort()
    np_f_train = f(np_x_train)

    # Solve the ODE
    ode = ODE.solve_ode(np_x_train, np_f_train, u_a, u_b)
    u_trained = ode(torch.tensor(np_x_train, dtype=torch.float32).view(-1, 1)).detach().numpy().flatten()

    # Plot the solution
    import matplotlib.pyplot as plt

    plt.plot(np_x_train, u_trained, linewidth=2, label='Trained solution', color='b')
    plt.plot(np_x_train, u_solution(np_x_train), linewidth=2, linestyle='--', label='Exact solution', color='r')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Solution of du/dx = 4x^3')
    plt.grid(True)
    plt.legend()
    plt.show()
