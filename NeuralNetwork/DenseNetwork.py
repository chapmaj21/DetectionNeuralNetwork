import numpy as np


class Relu:
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, grad, hidden):
        # Gradient passes only where the unit fired (output > 0 exactly when input > 0).
        return grad * (hidden > 0)


class Dense:
    """Fully connected layer: output = activation(x @ W + b).

    Works with dense NumPy arrays or SciPy sparse matrices as input.
    """

    def __init__(self, input_size, output_size, bias=True, activation=True, dtype=np.float32):
        self.add_bias = bias
        self.add_activation = activation
        self.activation = Relu()

        # Cached by forward() for use in backward(); grads are set by backward().
        self.hidden = None
        self.prev_hidden = None
        self.grads = None

        # He initialisation: N(0, 2 / fan_in), designed for ReLU layers.
        # Uses NumPy's global random state, seeded once in the training script.
        self.weights = (
            np.random.randn(input_size, output_size)
            * np.sqrt(2 / input_size)
        ).astype(dtype)
        self.bias = np.zeros((1, output_size), dtype=dtype)

    def forward(self, x):
        # Stored by reference: nothing modifies these arrays in place, so no copies are needed.
        self.prev_hidden = x
        x = x @ self.weights
        if self.add_bias:
            x += self.bias

        if self.add_activation:
            x = self.activation.forward(x)
        self.hidden = x
        return x

    def params(self):
        return [self.weights, self.bias] if self.add_bias else [self.weights]

    def backward(self, grad, lambda_reg=0.0, calculate_prev_grad=True):
        """Compute parameter gradients (stored in self.grads) and return dL/d(input).

        This only computes gradients; an optimiser such as SGD applies them.
        """
        if self.add_activation:
            grad = self.activation.backward(grad, self.hidden)

        batch_size = self.prev_hidden.shape[0]

        # Parameter gradients averaged across batch
        w_grad = self.prev_hidden.T @ grad / batch_size
        b_grad = np.mean(grad, axis=0, keepdims=True)

        # L2 regularisation (weights only)
        w_grad += lambda_reg * self.weights

        # Same order as params()
        self.grads = [w_grad, b_grad] if self.add_bias else [w_grad]

        # Gradient passed backwards is NOT averaged
        if calculate_prev_grad:
            return grad @ self.weights.T
        return None


class SGD:
    """Plain stochastic gradient descent: p <- p - lr * gradient.

    Kept separate from the layers so the update rule can be changed
    (e.g. to momentum) without editing any layer code.
    """

    def __init__(self, lr):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            for param, grad in zip(layer.params(), layer.grads):
                param -= self.lr * grad