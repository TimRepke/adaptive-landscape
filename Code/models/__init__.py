from abc import ABC, abstractmethod
from typing import Callable, Iterable


class Model(ABC):
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def fit_data(cls, data=None, input_file=None, params=None):
        pass

    @classmethod
    @abstractmethod
    def fit_intervals(cls, strategy: Callable, data: Iterable, params):
        pass

    def export(self):
        pass
