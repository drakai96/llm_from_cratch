from abc import ABC, abstractmethod

import numpy as np
from loguru import logger


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


class CrossEntropyLoss(Loss):
    def forward(self, predicts, labels):
        # Store predictions and labels for backward calculation
        self.predicts = np.clip(predicts, 1e-12, 1 - 1e-12)  # Avoid log(0) by clipping
        self.labels = labels

        # Compute cross-entropy loss
        value = -np.mean(np.sum(labels * np.log(self.predicts), axis=1))
        logger.debug(f"Cross-Entropy Loss: {value}")
        return value

    def backward(self, x):
        # Gradient calculation for multi-class logistic regression
        n = len(self.predicts)
        dw = -(1 / n) * (self.labels - self.predicts).T @ x
        db = np.mean(-(1 / n) * (self.labels - self.predicts), axis=0)
        return dw, db


class CrossEntropy(Loss):

    def __init__(self):
        super().__init__()

    def forward(self, predicts, labels):
        self.labels = labels
        self.predicts = np.clip(predicts, 1e-12, 1 - 1e-12)
        cross_entropy = -np.sum(self.labels * np.log(self.predicts), axis=1)
        # logger.debug(f"Cross-Entropy Loss: {cross_entropy}")
        return np.mean(cross_entropy)

    def backward(self, x):
        n = len(self.labels)
        dw = 1 / n * (self.predicts - self.labels).T @ x
        db = np.mean(1 / n * (self.predicts - self.labels), axis=0)
        return dw, db

    def __call__(self, predicts, labels):
        self.forward(predicts=predicts, labels=labels)
