import numpy as np
from models.lr import LinearRegression, LogisticRegression1


class Optimizer:
    def __init__(self, model: LinearRegression, learning_rate):
        self.model = model
        self.learning_rate = learning_rate

    def step(self, dw, db):
        if self.model.bias:
            self.model.bias -= np.dot(self.learning_rate, db)

        self.model.weight -= np.dot(self.learning_rate, dw.sum(axis=0))


class OptimizerLg:
    def __init__(self, model: LogisticRegression1, learning_rate):
        self.model = model
        self.learning_rate = learning_rate

    def step(self, dw, db):
        # Update weights
        self.model.weights -= self.learning_rate * dw.T

        # Update bias if it exists
        if self.model.use_bias and db is not None:
            self.model.bias -= self.learning_rate * np.array(db).T
