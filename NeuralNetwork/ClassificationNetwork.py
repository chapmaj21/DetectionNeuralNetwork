import copy
import numpy as np
import pandas as pd

from DenseNetwork import Dense

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def NLL(y, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return float(np.mean(-(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))))


class ClassificationNet:
    def __init__(self, input_size):
        self.layers = [
            Dense(input_size, 16, activation=True),
            # Dense(32, 8, activation=True),
            Dense(16, 1, activation=False)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, grad, lr, lambda_reg):
        for i in range(len(self.layers) - 1, -1, -1):
            grad = self.layers[i].backward(grad, lr, lambda_reg, calculate_prev_grad=(i != 0))

    def get_params(self):
        return [(layer.weights.copy(), layer.bias.copy()) for layer in self.layers]

    def set_params(self, params):
        for layer, (weights, bias) in zip(self.layers,params):
            layer.weights = weights.copy()
            layer.bias = bias.copy()


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

np.random.seed(0)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

data = pd.read_csv("../data/train.csv")
data = data.rename(columns={"comment_text": "comment"})
data["label"] = data["toxic"].astype(float)

data, _ = train_test_split(
    data,
    train_size=30000,
    random_state=0,
    stratify=data["label"]
)


# --------------------------------------------------
# Train / validation / test split
#
# 70% training
# 15% validation
# 15% test
# --------------------------------------------------

train_valid_data, test_data = train_test_split(
    data,
    test_size=0.15,
    random_state=0,
    stratify=data["label"]
)

train_data, valid_data = train_test_split(
    train_valid_data,
    test_size=0.1765,
    random_state=0,
    stratify=train_valid_data["label"]
)


print("Training examples:", len(train_data))
print("Validation examples:", len(valid_data))
print("Test examples:", len(test_data))


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    max_features=20000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True,
    dtype=np.float32
)
#--------------------------------------------------
# Split data into training, validation and test sets
#--------------------------------------------------

train_x = vectorizer.fit_transform(train_data["comment"])
valid_x = vectorizer.transform(valid_data["comment"])
test_x = vectorizer.transform(test_data["comment"])

train_y = train_data[["label"]].to_numpy(dtype=np.float32)
valid_y = valid_data[["label"]].to_numpy(dtype=np.float32)
test_y = test_data[["label"]].to_numpy(dtype=np.float32)

print("Number of TF-IDF features:", train_x.shape[1])


# --------------------------------------------------
# Initialise network
# --------------------------------------------------

net = ClassificationNet(train_x.shape[1])

# --------------------------------------------------
# Hyperparameters
# --------------------------------------------------

lr = 0.003
epochs = 2000
batch_size = 512
lambda_reg = 1e-4

# Validation is checked every 10 epochs.
validation_interval = 10

# Stop if validation loss has not improved
# for this many epochs.
patience = 100

# --------------------------------------------------
# Early stopping variables
# --------------------------------------------------

best_valid_loss = float("inf")
best_params = None
best_epoch = 0
epochs_without_improvement = 0

# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(epochs):
    train_loss = 0.0

    # Shuffle training examples each epoch
    indices = np.random.permutation(
        train_x.shape[0]
    )
    # ----------------------------------------------
    # Mini-batch gradient descent
    # ----------------------------------------------
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        x = train_x[batch_idx]
        target = train_y[batch_idx]

        # Forward pass
        logits = net.forward(x)
        prediction = sigmoid(logits)

        # Calculate batch loss
        train_loss += (NLL(target, prediction)* len(batch_idx))

        # BCE + sigmoid derivative
        #
        # dL / d(logit)
        grad = prediction - target

        # Backpropagation
        net.backward(grad, lr, lambda_reg)

    train_loss /= train_x.shape[0]


    # ----------------------------------------------
    # Validation
    # ----------------------------------------------

    if epoch % validation_interval == 0:
        valid_logits = net.forward(valid_x)
        valid_prediction = sigmoid(valid_logits)
        valid_loss = NLL(valid_y,valid_prediction)
        print(
            f"Epoch {epoch:4d} | "
            f"train loss: {train_loss:.4f} | "
            f"valid loss: {valid_loss:.4f}"
        )

        # ------------------------------------------
        # Save best model
        # ------------------------------------------

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_params = net.get_params()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += (validation_interval)

        # ------------------------------------------
        # Early stopping
        # ------------------------------------------

        if (epochs_without_improvement >= patience):
            print()
            print(f"Early stopping at epoch {epoch}")
            break


# --------------------------------------------------
# Restore best network
# --------------------------------------------------

if best_params is not None:
    net.set_params(best_params)

print()
print(f"Best validation loss: " f"{best_valid_loss:.4f}")
print(f"Best epoch: {best_epoch}")


# --------------------------------------------------
# Find best classification threshold
#
# IMPORTANT:
# Threshold is chosen using validation data only.
# --------------------------------------------------

valid_prob = sigmoid(net.forward(valid_x))

best_threshold = 0.5
best_valid_f1 = 0.0

for threshold in np.arange(0.01, 0.81, 0.01):
    predictions = (valid_prob >= threshold).astype(int)
    score = f1_score(valid_y, predictions, zero_division=0)
    if score > best_valid_f1:
        best_valid_f1 = score
        best_threshold = threshold

print()
print(f"Best threshold: " f"{best_threshold:.2f}")
print(f"Validation F1: " f"{best_valid_f1:.4f}")


# --------------------------------------------------
# Validation metrics
# --------------------------------------------------

valid_predictions = (
    valid_prob >= best_threshold
).astype(int)


print()
print("VALIDATION RESULTS")
print("------------------")

print("Accuracy:", accuracy_score(valid_y, valid_predictions))
print("Precision:", precision_score(valid_y, valid_predictions, zero_division=0))
print("Recall:", recall_score(valid_y, valid_predictions, zero_division=0))
print("F1:", f1_score(valid_y, valid_predictions, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(valid_y, valid_predictions))


# --------------------------------------------------
# Final evaluation
# --------------------------------------------------

test_prob = sigmoid(net.forward(test_x))
test_predictions = (test_prob >= best_threshold).astype(int)

print()
print("TEST RESULTS")
print("------------")

print("Accuracy:", accuracy_score(test_y, test_predictions))
print("Precision:", precision_score(test_y, test_predictions, zero_division=0))
print("Recall:", recall_score(test_y, test_predictions, zero_division=0))
print("F1:", f1_score(test_y, test_predictions, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(test_y, test_predictions))