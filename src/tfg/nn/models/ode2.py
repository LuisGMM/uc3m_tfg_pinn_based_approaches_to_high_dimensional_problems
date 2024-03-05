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
            nn.Linear(64, 64),  # Additional layer to handle second derivative
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def solve_ode(np_x_train, np_f_train, u_a, u_b, du_a, du_b, num_epochs=10000, lr=0.01):
        model = ODE()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        x_train = torch.tensor(np_x_train, dtype=torch.float32, requires_grad=True).view(-1, 1)
        f_train = torch.tensor(np_f_train, dtype=torch.float32).view(-1, 1)

        for _ in range(num_epochs):
            optimizer.zero_grad()
            u_train = model(x_train)

            # Compute first and second derivatives of u_train with respect to x_train
            u_grad = torch.autograd.grad(
                u_train,  # Sum along appropriate axis to ensure shape compatibility
                x_train,
                grad_outputs=torch.ones_like(u_train),
                create_graph=True,
                retain_graph=True,
            )[0]
            u_grad2 = torch.autograd.grad(
                u_grad,
                x_train,
                grad_outputs=torch.ones_like(u_grad),
                create_graph=True,
                retain_graph=True,
            )[0]

            # Compute boundary conditions
            u_a_pred = model(torch.tensor([[a]], dtype=torch.float32))
            u_b_pred = model(torch.tensor([[b]], dtype=torch.float32))
            du_a_pred = u_grad[0]
            du_b_pred = u_grad[-1]

            # Compute loss using u_grad2 and boundary conditions
            # loss = (
            #     criterion(u_grad2, f_train)  # Reshape f_train to match the shape of u_grad2
            #     + criterion(u_a_pred, torch.tensor([[u_a]], dtype=torch.float32))
            #     + criterion(u_b_pred, torch.tensor([[u_b]], dtype=torch.float32))
            #     + criterion(du_a_pred, torch.tensor([[du_a]], dtype=torch.float32))
            #     + criterion(du_b_pred, torch.tensor([[du_b]], dtype=torch.float32))
            # )
            loss = (
                criterion(u_grad2, f_train)
                + criterion(u_a_pred, torch.tensor([[u_a]], dtype=torch.float32).expand_as(u_a_pred))
                + criterion(u_b_pred, torch.tensor([[u_b]], dtype=torch.float32).expand_as(u_b_pred))
                + criterion(du_a_pred, torch.tensor([du_a], dtype=torch.float32).expand_as(du_a_pred))
                + criterion(du_b_pred, torch.tensor([du_b], dtype=torch.float32).expand_as(du_b_pred))
            )

            loss.backward()
            optimizer.step()

        return model


if __name__ == '__main__':
    # Define the interval [a, b] and the boundary conditions u(a), u(b), du(a)/dx, du(b)/dx
    a = 0
    b = 2.0

    def u_solution(x):
        return 10 * x**5

    def dudx_solution(x):
        return 10 * 5 * x**4

    def f(x):
        return 10 * 5 * 4 * x**3

    u_a = u_solution(a)
    u_b = u_solution(b)
    du_a = dudx_solution(a)
    du_b = dudx_solution(b)

    np_x_train = np.random.uniform(a, b, 1000)
    np_x_train.sort()
    np_f_train = f(np_x_train)

    # Solve the ODE
    ode = ODE.solve_ode(np_x_train, np_f_train, u_a, u_b, du_a, du_b)
    u_trained = ode(torch.tensor(np_x_train, dtype=torch.float32).view(-1, 1)).detach().numpy().flatten()

    # Plot the solution
    import matplotlib.pyplot as plt

    plt.plot(np_x_train, u_trained, linewidth=2, label='Trained solution', color='b')
    plt.plot(np_x_train, u_solution(np_x_train), linewidth=2, label='True solution', color='r', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Solution of du^2/d^2x = 12x^2')
    plt.grid(True)
    plt.legend()
    plt.show()
