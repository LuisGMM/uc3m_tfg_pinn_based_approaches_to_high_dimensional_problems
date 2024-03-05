
# Tasks

## Weekly tasks

### Week 1

[x] Create a NN to approximate f(x)=cos(nx)
[x] Create a NN to approximate something with more dimensions, f(x, y)=x^1 + y^2

### Week 2

[x] Add a variable learning rate to reduce the learing rate as the loss decreases
[x] Explore whether it is possible to use at each epoch a different subset (maybe
a randomnly and uniformly distributed) of the training data.
[x] Try to do the above with a function in higher dimensions, that is, maybe a
paraboloid with multiple inputs
f(r, s, t, u, v, w) = r*s + cos(t) + sin(2u) + v^3 + exp^w
[x] Learn to differentiate the output of the nn with respect to the different inputs

### Week 3

[x] Test the previous week against some data and check its validity.
[x] Plot also over an R^6 line (line vs nn(line))
[] Implement a first order ODE:

```bash
u = exp(x * cos(x)^2)

du/dx = f
input (compute it): f
loss: MSE(d(nn)/dx - f) + MSE(nn(a) - u(a)) + MSE(nn(b) - u(b))

in some interval [a, b]
```

[] Extra: Second order ODE or 2 variable PDE

### Week 4

[] Implement a first order ODE:
[] Extra: Second order ODE or 2 variable PDE


### Week 5

[x] Implement a first order ODE:
[x] Implement Second order ODE
[x] Implement 2 variable PDE
