import abc
import os
import sys
from abc import ABC, abstractmethod
from ast import literal_eval
from typing import List, Tuple, Union, Dict, Any

import numpy as np


class BaseDetector(ABC):
    @abstractmethod
    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def load(self): ...

    @abstractmethod
    def detect(self, image: np.ndarray, roi: Union[List[Tuple[int, int, int, int]], None] = None) \
            -> List[Tuple[int, int, int, int]]: ...


# Auxiliary functions
def get_detector(name: str) -> abc.ABCMeta:
    """ Load the detector dynamically in runtime.

    Args:
        name: String with the name of the detector in format "module.class".

    Returns: A metaclass to be used to instantiate a detector.
    """
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    groups = name.split('.')
    module = ".".join(groups[:-1])
    mod = __import__(module)
    for comp in groups[1:]:
        mod = getattr(mod, comp)

    if isinstance(mod, abc.ABCMeta):
        return mod

    raise Exception("The name '" + name + "' is not a valid detector class. Check your config file.")


def get_field(config: Dict, field: str, eval_field: bool = False) -> Any:
    """ Get a field from a dictionary. If the field does not exist raises an Exception.

    Args:
        config: Dictionary to get the field value from.
        field: A string with the name of the key in the dictionary config.
        eval_field: Whether the field requires a literal evaluation.

    Returns:
        The value in the dict corresponding to the key named with field variable.
    """
    value = config.get(field)
    if value:
        if eval_field:
            return literal_eval(value)
        return value
    raise Exception("Field '" + field + "' not found in the config file. Check your config file.")
