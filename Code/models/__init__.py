from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit_interval(self):
        pass

    def export(self):
        pass
