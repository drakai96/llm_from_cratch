from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    def __init__(self):
        self.predicts = None
        self.labels = None

    @abstractmethod
    def forward(self, predicts, labels):
        raise NotImplementedError

    @abstractmethod
    def backward(self, x):
        raise NotImplementedError

    def __call__(self, predicts, labels):
        self.forward(predicts, labels)


class MSELoss(Loss):
    def forward(self, predicts, labels):
        self.predicts = predicts
        self.labels = labels
        value = np.mean((labels - predicts) ** 2)
        return value

    def backward(self, x):
        n = len(self.predicts)
        dw = -(2 / n) * (self.labels - self.predicts)[:, np.newaxis] * x
        db = np.mean(-(2 / n) * (self.labels - self.predicts))

        return dw, db
