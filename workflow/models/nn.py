from abc import ABC, abstractmethod


class Module(ABC):

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def _init_weight(self):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)
