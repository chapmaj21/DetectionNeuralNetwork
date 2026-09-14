import numpy as np
import pandas as pd

from DenseNetwork import Dense
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def NLL(y, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    return float(np.mean(
        -(y * np.log(y_pred)
          + (1 - y) * np.log(1 - y_pred))
    ))


class ClassificationNet:
    def __init__(self, input_size):
        self.output = Dense(
            input_size,
            1,
            activation=False
        )

    def forward(self, x):
        return self.output.forward(x)

    def backward(self, grad, lr):
        self.output.backward(grad, lr)

np.random.seed(0)

data = pd.read_csv(
    "../data/Ethos_Dataset_Binary.csv",
    delimiter=";"
)

data["label"] = (data["isHate"] >= 0.5).astype(float)

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

vectorizer = TfidfVectorizer(
    lowercase=True,
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

train_x = vectorizer.fit_transform(
    train_data["comment"]
).toarray()

valid_x = vectorizer.transform(
    valid_data["comment"]
).toarray()

test_x = vectorizer.transform(
    test_data["comment"]
).toarray()

train_y = train_data[["label"]].to_numpy()
valid_y = valid_data[["label"]].to_numpy()
test_y = test_data[["label"]].to_numpy()

net = ClassificationNet(train_x.shape[1])

lr = 0.03
epochs = 1000
batch_size = 32

for epoch in range(epochs):
    train_loss = 0.0
    indices = np.random.permutation(len(train_x))

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        x = train_x[batch_idx]
        target = train_y[batch_idx]
        prediction = sigmoid(net.forward(x))
        train_loss += NLL(target, prediction) * len(batch_idx)
        grad = prediction - target
        net.backward(grad, lr)

    train_loss /= len(train_x)

    valid_prediction = sigmoid(net.forward(valid_x))
    valid_loss = NLL(valid_y, valid_prediction)

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:3d} | "
            f"train loss: {train_loss:.4f} | "
            f"valid loss: {valid_loss:.4f}"
        )


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

valid_prob = sigmoid(net.forward(valid_x))
valid_predictions = (valid_prob >= 0.43).astype(int)

print("Accuracy:", accuracy_score(valid_y, valid_predictions))
print("Precision:", precision_score(valid_y, valid_predictions, zero_division=0))
print("Recall:", recall_score(valid_y, valid_predictions, zero_division=0))
print("F1:", f1_score(valid_y, valid_predictions, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(valid_y, valid_predictions))