from numpy import array
from pandas import Series, DataFrame
from typing import Union, List, Dict, Tuple, TypeVar, Generator

T = TypeVar("T")

YielderGenerator = Generator[T, None, None]

KerasInputOutput = Union[List[array], array]
SimpleTypeTexts = Union[List[str], Series]
MultiTypeTexts = Union[Dict[str, List[str]], DataFrame]
TextsInput = Union[SimpleTypeTexts, MultiTypeTexts]
ModelFitTuple = Tuple[TextsInput, KerasInputOutput]
KerasModelFitTuple = Tuple[KerasInputOutput, KerasInputOutput]
