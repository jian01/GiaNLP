from typing import Union, List, Dict, Tuple, TypeVar, Generator, cast, overload

from numpy import array, ndarray
from pandas import Series, DataFrame

T = TypeVar("T")

YielderGenerator = Generator[T, None, None]

KerasInputOutput = Union[List[array], array]
SimpleTypeTexts = Union[List[str], Series]
MultiTypeTexts = Union[Dict[str, List[str]], DataFrame]
TextsInput = Union[SimpleTypeTexts, MultiTypeTexts]
ModelFitTuple = Tuple[TextsInput, KerasInputOutput]
KerasModelFitTuple = Tuple[KerasInputOutput, KerasInputOutput]


class TextsInputWrapper:
    """
    Wrapper for objects of type TextsInput
    """

    texts: Union[List[str], Dict[str, List[str]]]

    def __init__(self, texts_input: TextsInput):
        """

        :param texts_input: the text inputs
        :raises ValueError: if text inputs is a TextsInputWrapper object
        """
        if isinstance(texts_input, TextsInputWrapper):
            raise ValueError("Can't wrap an already wrapped text input.")

        if isinstance(texts_input, Series):
            self.texts = texts_input.values.tolist()
        elif isinstance(texts_input, DataFrame):
            self.texts = texts_input.to_dict("list")
        else:
            self.texts = texts_input

    def is_multi_text(self) -> bool:
        """
        Returns True if input is multi-text

        :return: a boolean indicating if it's multi-text
        """
        return isinstance(self.texts, dict)

    @overload
    def __getitem__(self, key: Union[slice, array, List[int]]) -> "TextsInputWrapper":
        ...

    @overload
    def __getitem__(self, key: Union[int, str]) -> Union[str, List[str]]:
        ...

    def __getitem__(self, key: Union[int, str, slice, array, List[int]]) -> Union[str, List[str], "TextsInputWrapper"]:
        """
        Returns the result of indexing the texts.
        If the text is from multiple types, its indexed by string key and
        returns a list of texts.
        If the text is just one type of text, its indexed by number.
        If the key is a slice, a slice is made to the lists of texts.
        If the key is a numpy array or list of ints, reorders and slices all the texts.

        :param key: the key for indexing
        :return: a string or list of strings or a new TextsInputWrapper when sliced
        :raises KeyError:
            - if the text input is not multiple and the key is str
            - if the text input is multiple and the key is int
            - if the key is not present or the type of the key is not valid
        """
        if isinstance(key, ndarray) or isinstance(key, list):
            if isinstance(key, ndarray):
                key = key.tolist()
            if self.is_multi_text():
                self.texts = cast(Dict[str, List[str]], self.texts)
                return TextsInputWrapper({k: [v[i] for i in key] for k, v in self.texts.items()})
            return TextsInputWrapper([self[i] for i in key])
        if isinstance(key, slice):
            if self.is_multi_text():
                self.texts = cast(Dict[str, List[str]], self.texts)
                return TextsInputWrapper({k: v[key] for k, v in self.texts.items()})
            return TextsInputWrapper(self.texts[key])
        if isinstance(key, str):
            if not self.is_multi_text():
                raise KeyError("Can't index a simple text input by str type")
            return self.texts[key]
        if isinstance(key, int):
            if self.is_multi_text():
                raise KeyError("Can't index a multi text input by int type")
            return self.texts[key]
        raise KeyError("The type of the key is not valid")

    def __add__(self, other) -> "TextsInputWrapper":
        """
        Adds two text input wrappers

        :param other: the other text input wrapper
        :return: the result of adding texts of both inputs
        :raises ValueError:
            - if both text inputs are multiple and keys don't match
            - if one text input is multiple and the other is not
        """
        if self.is_multi_text() ^ other.is_multi_text():
            raise ValueError("One of the text inputs is multiple and the other is not.")
        if self.is_multi_text():
            self.texts = cast(Dict[str, List[str]], self.texts)
            if set(self.texts.keys()) != set(other.to_texts_inputs().keys()):
                raise ValueError("Key's for multi-text inputs do not match")
            return TextsInputWrapper({k: v + other[k] for k, v in self.texts.items()})
        return TextsInputWrapper(self.texts + other.to_texts_inputs())

    def __len__(self) -> int:
        """
        Computes the length of text inputs
        :return: the length
        """
        if isinstance(self.texts, dict):
            return len(self.texts[list(self.texts.keys())[0]])
        return len(self.texts)

    def to_texts_inputs(self) -> TextsInput:
        """
        Transform to text inputs type

        :return: a text input type object
        """
        return self.texts.copy()
