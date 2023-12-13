import abc
import numpy as np
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

class BaseDetector(ABC, metaclass=abc.ABCMeta):
    config: Incomplete
    @abstractmethod
    def __init__(self, config: Dict): ...
    @abstractmethod
    def load(self): ...
    @abstractmethod
    def detect(self, image: np.ndarray, roi: Union[List[Tuple[int, int, int, int]], None] = ...) -> List[Tuple[int, int, int, int]]: ...

def get_detector(name: str) -> abc.ABCMeta: ...
def get_field(config: Dict, field: str, eval_field: bool = ...) -> Any: ...
