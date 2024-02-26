# u = exp(x * cos(x)^2)
#
# du/dx = f
# input (compute it): f
# loss: MSE(d(nn)/dx - f) + MSE(nn(a) - u(a)) + MSE(nn(b) - u(b))
# in some interval [a, b]
# also add u(a), u(b) to the loss
