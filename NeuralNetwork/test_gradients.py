"""Numerical gradient check for the hand-written backprop.

Run:  python NeuralNetwork/test_gradients.py   (or: pytest NeuralNetwork)

For every weight and bias, compares the analytic gradient that backward()
stores in layer.grads with the finite difference
(L(p + eps) - L(p - eps)) / (2 * eps).
"""

import numpy as np
from scipy import sparse

from ClassificationNetwork import NLL, ClassificationNet, sigmoid
from DenseNetwork import SGD

LAMBDA_REG = 1e-3


def params_of(net):
    return [p for layer in net.layers for p in layer.params()]


def total_loss(net, x, y):
    # Mean BCE plus the L2 term whose gradient is lambda * W.
    l2 = sum(np.sum(layer.weights ** 2) for layer in net.layers)
    return NLL(y, sigmoid(net.forward(x))) + 0.5 * LAMBDA_REG * l2


def numeric_grads(net, x, y, eps):
    grads = []
    for param in params_of(net):
        grad = np.zeros_like(param)
        for idx in np.ndindex(param.shape):
            original = param[idx]
            param[idx] = original + eps
            loss_plus = total_loss(net, x, y)
            param[idx] = original - eps
            loss_minus = total_loss(net, x, y)
            param[idx] = original
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        grads.append(grad)
    return grads


def analytic_grads(net, x, y):
    prediction = sigmoid(net.forward(x))
    net.backward(prediction - y, lambda_reg=LAMBDA_REG)
    return [g for layer in net.layers for g in layer.grads]


def make_net(x):
    np.random.seed(1)
    y = (np.random.rand(x.shape[0], 1) < 0.3).astype(np.float64)
    net = ClassificationNet(x.shape[1], hidden_sizes=[6, 4], dtype=np.float64)
    return net, y


def check_gradients(x, eps=1e-6, tol=1e-6):
    net, y = make_net(x)
    numeric = numeric_grads(net, x, y, eps)
    analytic = analytic_grads(net, x, y)

    for a, n in zip(analytic, numeric):
        rel_error = np.linalg.norm(a - n) / (np.linalg.norm(a) + np.linalg.norm(n) + 1e-12)
        assert rel_error < tol, f"gradient mismatch, relative error {rel_error:.2e}"


def test_gradients_dense_input():
    np.random.seed(0)
    check_gradients(np.random.randn(20, 8))


def test_gradients_sparse_input():
    # The real model is fed SciPy sparse TF-IDF matrices, so check that path too.
    x = sparse.random(20, 30, density=0.2, format="csr", random_state=0, dtype=np.float64)
    check_gradients(x)


def test_sgd_step():
    # SGD should move every parameter by exactly -lr * gradient.
    np.random.seed(0)
    x = np.random.randn(20, 8)
    net, y = make_net(x)
    grads = analytic_grads(net, x, y)
    before = [p.copy() for p in params_of(net)]

    SGD(lr=0.5).step(net.layers)

    for b, a, g in zip(before, params_of(net), grads):
        assert np.allclose(a, b - 0.5 * g)


if __name__ == "__main__":
    test_gradients_dense_input()
    test_gradients_sparse_input()
    test_sgd_step()
    print("Gradient checks passed.")