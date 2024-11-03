import numpy as np
from models.lr import LinearRegression


class Optimizer:
    def __init__(self, model: LinearRegression, learning_rate):
        self.model = model
        self.learning_rate = learning_rate

    def step(self, dw, db):
        if self.model.bias:
            self.model.bias -= np.dot(self.learning_rate, db)

        self.model.weight -= np.dot(self.learning_rate, dw.sum(axis=0))
