from abc import ABC, abstractmethod, abstractproperty
from typing import Union, List, Iterable, Dict


class DataReader(ABC):
    def __init__(self, raw_data_dir):
        self.raw_data_dir = raw_data_dir

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def data_labels(self) -> List[Union[int, str]]:
        raise NotImplementedError

    @abstractmethod
    def generate_data(self):
        raise NotImplementedError

    @abstractmethod
    def get_data(self):
        raise NotImplementedError

    @abstractmethod
    def generate_labels(self):
        raise NotImplementedError

    @abstractmethod
    def get_labels(self):
        raise NotImplementedError
