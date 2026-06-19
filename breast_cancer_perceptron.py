import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = load_breast_cancer()

X = data.data
y = data.target


#
# df = pd.read_csv("breast_cancer.csv")
#
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values


print("Features Shape:", X.shape)
print("Labels Shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


class Perceptron:

    def __init__(self, lr=0.001, epochs=100):

        self.lr = lr
        self.epochs = epochs

    def step_function(self, x):

        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):

        samples, features = X.shape

        self.weights = np.zeros(features)

        self.bias = 0

        for epoch in range(self.epochs):

            for index, x_i in enumerate(X):

                linear_output = np.dot(
                    x_i,
                    self.weights
                ) + self.bias

                prediction = self.step_function(
                    linear_output
                )

                update = self.lr * (
                    y[index] - prediction
                )

                self.weights += update * x_i

                self.bias += update

    def predict(self, X):

        linear_output = np.dot(
            X,
            self.weights
        ) + self.bias

        return self.step_function(
            linear_output
        )


model = Perceptron(
    lr=0.001,
    epochs=100
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)
