"""Toxic comment classifier: TF-IDF features + a small neural network in NumPy.

Run from anywhere:  python NeuralNetwork/ClassificationNetwork.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from DenseNetwork import SGD, Dense


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "train.csv"

SEED = 0
SUBSAMPLE = 150000           # comments to use, out of 159,571 (must be smaller than that); 30000 for quick runs
HIDDEN_SIZES = [16]          # e.g. [32, 8] for two hidden layers

# TF-IDF inputs are sparse (~66 non-zero values per comment), so each weight gets
# few, small gradients and needs a much larger step than dense data would.
LR = 0.1
EPOCHS = 2000
BATCH_SIZE = 512
LAMBDA_REG = 1e-4

VALIDATION_INTERVAL = 10     # check validation loss every this many epochs
PATIENCE = 100               # stop after this many epochs without improvement


# --------------------------------------------------
# Activation and loss
# --------------------------------------------------

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def NLL(y, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return float(np.mean(-(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))))


# --------------------------------------------------
# Model
# --------------------------------------------------

class ClassificationNet:
    def __init__(self, input_size, hidden_sizes=(16,), dtype=np.float32):
        # Hidden layers use ReLU; the output layer returns a logit (sigmoid applied outside).
        sizes = [input_size, *hidden_sizes, 1]
        n_layers = len(sizes) - 1
        self.layers = [
            Dense(sizes[i], sizes[i + 1], activation=(i < n_layers - 1), dtype=dtype)
            for i in range(n_layers)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, lambda_reg):
        """Compute gradients for every layer. An optimiser applies them afterwards."""
        for i in range(len(self.layers) - 1, -1, -1):
            # The first layer's input gradient is never used, so skip computing it.
            grad = self.layers[i].backward(grad, lambda_reg, calculate_prev_grad=(i != 0))

    def get_params(self):
        return [(layer.weights.copy(), layer.bias.copy()) for layer in self.layers]

    def set_params(self, params):
        for layer, (weights, bias) in zip(self.layers, params):
            layer.weights = weights.copy()
            layer.bias = bias.copy()


# --------------------------------------------------
# Data
# --------------------------------------------------

def load_data():
    data = pd.read_csv(DATA_PATH)
    data = data.rename(columns={"comment_text": "comment"})
    data["label"] = data["toxic"].astype(float)

    data, _ = train_test_split(
        data,
        train_size=SUBSAMPLE,
        random_state=SEED,
        stratify=data["label"]
    )
    return data


def split_data(data):
    """70% training / 10% validation / 5% threshold tuning / 15% test, all stratified by label.

    Validation picks the epoch (early stopping) and a separate tuning set picks the
    decision threshold, so neither choice is fitted to the data used to judge the other.
    """
    train_valid_data, test_data = train_test_split(
        data,
        test_size=0.15,
        random_state=SEED,
        stratify=data["label"]
    )
    # 17.65% of the remaining 85% is 15% of the whole dataset (0.15 / 0.85 ≈ 0.1765).
    train_data, holdout_data = train_test_split(
        train_valid_data,
        test_size=0.1765,
        random_state=SEED,
        stratify=train_valid_data["label"]
    )
    # Split that 15% into 10% validation and 5% threshold tuning.
    valid_data, tune_data = train_test_split(
        holdout_data,
        test_size=1 / 3,
        random_state=SEED,
        stratify=holdout_data["label"]
    )
    return train_data, valid_data, tune_data, test_data


def make_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        max_features=20000,
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
        dtype=np.float32
    )


def labels(df):
    return df[["label"]].to_numpy(dtype=np.float32)


# --------------------------------------------------
# Training
# --------------------------------------------------

def train(net, train_x, train_y, valid_x, valid_y):
    optimizer = SGD(LR)

    best_valid_loss = float("inf")
    best_params = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        # Shuffle training examples each epoch
        indices = np.random.permutation(train_x.shape[0])

        # Mini-batch gradient descent
        for start in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]
            x = train_x[batch_idx]
            target = train_y[batch_idx]

            prediction = sigmoid(net.forward(x))

            # BCE + sigmoid derivative: dL / d(logit) = prediction - target
            grad = prediction - target
            net.backward(grad, LAMBDA_REG)
            optimizer.step(net.layers)

        if epoch % VALIDATION_INTERVAL == 0:
            # Both losses are measured after the epoch with the same fixed weights,
            # so they are directly comparable.
            train_loss = NLL(train_y, sigmoid(net.forward(train_x)))
            valid_loss = NLL(valid_y, sigmoid(net.forward(valid_x)))
            print(f"Epoch {epoch:4d} | train loss: {train_loss:.4f} | valid loss: {valid_loss:.4f}")

            # Save the best model; stop if validation loss hasn't improved for PATIENCE epochs.
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                best_params = net.get_params()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += VALIDATION_INTERVAL

            if epochs_without_improvement >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Restore the best network
    if best_params is not None:
        net.set_params(best_params)

    print(f"\nBest validation loss: {best_valid_loss:.4f}")
    print(f"Best epoch: {best_epoch}")
    return net


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def choose_threshold(y, prob):
    """Pick the decision threshold with the best F1. Use the tuning set only, never test."""
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in np.arange(0.01, 0.81, 0.01):
        score = f1_score(y, (prob >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    return best_threshold, best_f1


def report(title, y, prob, threshold):
    predictions = (prob >= threshold).astype(int)
    print(f"\n{title}")
    print("-" * len(title))
    print("Accuracy:", accuracy_score(y, predictions))
    print("Precision:", precision_score(y, predictions, zero_division=0))
    print("Recall:", recall_score(y, predictions, zero_division=0))
    print("F1:", f1_score(y, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y, predictions))


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    np.random.seed(SEED)  # reproducible weight initialisation and shuffling

    data = load_data()
    train_data, valid_data, tune_data, test_data = split_data(data)
    print("Training examples:", len(train_data))
    print("Validation examples:", len(valid_data))
    print("Threshold tuning examples:", len(tune_data))
    print("Test examples:", len(test_data))

    # Fit the vocabulary and IDF on training data only, to avoid leakage.
    vectorizer = make_vectorizer()
    train_x = vectorizer.fit_transform(train_data["comment"])
    valid_x = vectorizer.transform(valid_data["comment"])
    tune_x = vectorizer.transform(tune_data["comment"])
    test_x = vectorizer.transform(test_data["comment"])
    train_y, valid_y = labels(train_data), labels(valid_data)
    tune_y, test_y = labels(tune_data), labels(test_data)
    print("Number of TF-IDF features:", train_x.shape[1])

    net = ClassificationNet(train_x.shape[1], HIDDEN_SIZES)
    net = train(net, train_x, train_y, valid_x, valid_y)

    tune_prob = sigmoid(net.forward(tune_x))
    threshold, tune_f1 = choose_threshold(tune_y, tune_prob)
    print(f"\nBest threshold (from tuning set): {threshold:.2f}")
    print(f"Tuning set F1: {tune_f1:.4f}")

    report("VALIDATION RESULTS", valid_y, sigmoid(net.forward(valid_x)), threshold)
    report("TEST RESULTS", test_y, sigmoid(net.forward(test_x)), threshold)


if __name__ == "__main__":
    main()