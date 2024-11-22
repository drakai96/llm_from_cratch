import numpy as np
from loguru import logger
from models import nn


class LinearRegression(nn.Module):
    def __init__(self, n_features, use_bias=True):
        super(LinearRegression).__init__()
        self.n_features = n_features
        self.use_bias = use_bias
        self._init_weight()

    def _init_weight(self):
        self.weight = np.random.randn(self.n_features)

        if self.use_bias:
            self.bias = np.random.randn()
        else:
            self.bias = None

        logger.debug(f"Init weight: {self.weight} and bias: {self.bias}")

    def forward(self, x):
        output = np.dot(x, self.weight)
        if self.bias:
            output += self.bias
        return output


class LogisticRegression1:
    def __init__(self, n_features, n_classes, use_bias=True):
        self.n_features = n_features
        self.n_classes = n_classes
        self.use_bias = use_bias
        self._init_weight()  # Ensure naming consistency here

    def _init_weight(self):  # Renamed from _init_weights to _init_weight
        # Initialize weights and biases
        self.weights = np.random.randn(self.n_features, self.n_classes)
        self.bias = np.random.randn(self.n_classes) if self.use_bias else None
        logger.debug(f"Init weights: {self.weights} and bias: {self.bias}")

    def softmax(self, x):
        # Softmax activation for multi-class probabilities
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        # Linear transformation followed by softmax
        linear_output = np.dot(x, self.weights)
        if self.use_bias:
            linear_output += self.bias
        return self.softmax(linear_output)

    def __call__(self, x):
        return self.forward(x)


# class LogisticRegression(nn.Module):

#     def __init__(self, n_feature, n_classes, use_bias=True):
#         self.n_feature = n_feature
#         self.n_classes = n_classes
#         self.use_bias = use_bias
#         self._init_weight()

#     def _init_weight(
#         self,
#     ):
#         self.weights = np.random.randn(self.n_feature, self.n_classes)
#         if self.use_bias:
#             self.bias = np.random.randn(self.n_classes)

#     def softmax(self, X):
#         """
#         This method use to calculate the soft max of matrix probality
#         """
#         exp = np.exp(X - np.max(X, keepdims=True, axis=1))
#         return exp / np.sum(X, axis=1, keepdims=True)

#     def forward(self, X):
#         linear_out = np.dot(X, self.weights)
#         if self.use_bias:
#             linear_out += self.bias
#         return self.softmax(linear_out)

#     def __call__(self, x):
#         return super().__call__(x)
