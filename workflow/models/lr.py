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
