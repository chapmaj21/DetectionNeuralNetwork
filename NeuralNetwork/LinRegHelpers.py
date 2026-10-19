"""Warm-up: linear regression by gradient descent, predicting tomorrow's max temperature.

Run from anywhere:  python NeuralNetwork/LinRegHelpers.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "clean_weather.csv"
PREDICTORS = ["tmax", "tmin", "rain"]
TARGET = "tmax_tomorrow"


def load_splits():
    data = pd.read_csv(DATA_PATH, index_col=0)
    data = data.ffill()

    # Chronological split: training (0-70%), validation (70-85%) and test (85-100%).
    # No shuffling, so the model is never trained on days after the ones it's judged on.
    train_end = int(0.7 * len(data))
    valid_end = int(0.85 * len(data))
    split_data = [
        data.iloc[:train_end],
        data.iloc[train_end:valid_end],
        data.iloc[valid_end:]
    ]
    return [
        (d[PREDICTORS].to_numpy(), d[[TARGET]].to_numpy())
        for d in split_data
    ]


# Creates initial weights & biases for the model
def init_params(num_predictors):
    np.random.seed(0)     # Initialises weights and biases the same every time the algo is run - same performance
    weights = np.random.rand(num_predictors, 1)
    biases = np.ones((1, 1))
    return [weights, biases]


# Calculates predictions for the x values using the weights & biases
def forward(params, x):
    weights, biases = params
    prediction = x @ weights + biases
    return prediction


# Calculates mean squared error
def mse(actual, pred):
    return np.mean((actual - pred) ** 2)


# Calculates the gradient of the loss w.r.t predictions (derivative of MSE).
# The true derivative is 2 * (pred - actual); the factor of 2 is absorbed into the learning rate.
def mse_grad(actual, pred):
    return (pred - actual)    # returns a vector


# Calculates how much weights & biases contributed to error + adjusts them accordingly
# For each weight:
#    calculate mean(feature × error)
#    multiply by learning rate
#    subtract from that weight
# For the bias:
#    calculate mean(error)
#    multiply by learning rate
#    subtract from the bias
def backward(params, x, lr, grad):
    w_grad = (x.T / x.shape[0]) @ grad         # x1*g, x2*g, x3*g
    b_grad = np.mean(grad, axis=0)
    params[0] -= w_grad * lr
    params[1] -= b_grad * lr
    return params


def main():
    (train_x, train_y), (valid_x, valid_y), _ = load_splits()

    lr = 1e-4
    epochs = 100
    params = init_params(train_x.shape[1])   # shape[1] gives how many features each data point has

    for i in range(epochs):
        predictions = forward(params, train_x)
        grad = mse_grad(train_y, predictions)
        params = backward(params, train_x, lr, grad)
        if i % 10 == 0:
            predictions = forward(params, valid_x)
            valid_loss = mse(valid_y, predictions)
            print(f"Epoch {i} loss: {valid_loss}")


if __name__ == "__main__":
    main()