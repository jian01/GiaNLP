from typing import Union, List, Dict, Tuple, TypeVar, Generator, cast, overload

import numpy as np
from numpy import ndarray
from pandas import Series, DataFrame

T = TypeVar("T")

YielderGenerator = Generator[T, None, None]

KerasInputOutput = Union[List[ndarray], ndarray]
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
    def __getitem__(self, key: Union[slice, ndarray, List[int]]) -> "TextsInputWrapper":
        ...

    @overload
    def __getitem__(self, key: Union[int, str]) -> Union[str, List[str]]:
        ...

    def __getitem__(
        self, key: Union[int, str, slice, ndarray, List[int]]
    ) -> Union[str, List[str], "TextsInputWrapper"]:
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
            key = cast(List[int], key)
            if self.is_multi_text():
                self.texts = cast(Dict[str, List[str]], self.texts)
                return TextsInputWrapper({k: [v[i] for i in key] for k, v in self.texts.items()})
            return TextsInputWrapper([self[i] for i in key])
        if isinstance(key, slice):
            key = cast(slice, key)
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

    def __add__(self, other: "TextsInputWrapper") -> "TextsInputWrapper":
        """
        Concatenates two text input wrappers

        :param other: the other text input wrapper
        :return: the result of adding texts of both inputs
        :raises ValueError:
            - if both text inputs are multiple and keys don't match
            - if one text input is multiple and the other is not
        """
        if self.is_multi_text() ^ other.is_multi_text():
            raise ValueError("One of the text inputs is multiple and the other is not.")
        other_text_inputs = other.to_texts_inputs()
        if self.is_multi_text():
            self.texts = cast(Dict[str, List[str]], self.texts)
            other_text_inputs = cast(Dict[str, List[str]], other_text_inputs)
            if set(self.texts.keys()) != set(other_text_inputs.keys()):
                raise ValueError("Key's for multi-text inputs do not match")
            return TextsInputWrapper({k: v + other_text_inputs[k] for k, v in self.texts.items()})
        self.texts = cast(List[str], self.texts)
        other_text_inputs = cast(List[str], other_text_inputs)
        return TextsInputWrapper(self.texts + other_text_inputs)

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


class ModelOutputsWrapper:
    """
    Wrapper for objects used as model outputs
    """

    keras_io: KerasInputOutput

    def __init__(self, keras_io: KerasInputOutput):
        """

        :param keras_io: the keras outputs
        :raises ValueError: if keras_io is a ModelOutputsWrapper object
        """
        if isinstance(keras_io, ModelOutputsWrapper):
            raise ValueError("Can't wrap an already wrapped model output.")
        self.keras_io = keras_io

    def __getitem__(self, key: Union[slice, ndarray, List[int]]) -> "ModelOutputsWrapper":
        """
        Returns the result of indexing the outputs.
        If the key is slice, array or list it retrieves those items for each of the outputs (multiple or not)

        :param key: the key for indexing
        :return: a ModelOutputsWrapper
        :raises KeyError: if the type of key is not valid or unexistent
        """
        if isinstance(key, ndarray) or isinstance(key, list):
            if isinstance(key, ndarray):
                key = key.tolist()
            key = cast(List[int], key)
            if isinstance(self.keras_io, list):
                self.keras_io = cast(List[ndarray], self.keras_io)
                return ModelOutputsWrapper([np.asarray([out[i] for i in key]) for out in self.keras_io])
            return ModelOutputsWrapper(np.asarray([self.keras_io[i] for i in key]))
        if isinstance(key, slice):
            key = cast(slice, key)
            if isinstance(self.keras_io, list):
                self.keras_io = cast(List[ndarray], self.keras_io)
                return ModelOutputsWrapper([out[key] for out in self.keras_io])
            return ModelOutputsWrapper(self.keras_io[key])
        raise KeyError("The type of the key is not valid")

    def __add__(self, other: "ModelOutputsWrapper") -> "ModelOutputsWrapper":
        """
        Concatenates two model outputs

        :param other: the other model output
        :return: the result of adding both model outputs
        :raises ValueError:
            - if one output is multiple and the other is not
            - if both are multiple outputs and amount of outputs don't match
        """
        other_model_outputs = other.to_model_outputs()
        if isinstance(other_model_outputs, list) ^ isinstance(self.keras_io, list):
            raise ValueError("Outputs types don't match")
        if isinstance(self.keras_io, list):
            if len(self.keras_io) != len(other_model_outputs):
                raise ValueError("The lengths of the outputs do not match")
            return ModelOutputsWrapper(
                [np.asarray(y1.tolist() + y2.tolist()) for y1, y2 in zip(self.keras_io, other_model_outputs)]
            )
        other_model_outputs = cast(ndarray, other_model_outputs)
        return ModelOutputsWrapper(np.asarray(self.keras_io.tolist() + other_model_outputs.tolist()))

    def __len__(self) -> int:
        """
        Computes the length of text inputs
        :return: the length
        """
        if isinstance(self.keras_io, list):
            return len(self.keras_io[0])
        return len(self.keras_io)

    def to_model_outputs(self) -> KerasInputOutput:
        """
        Transform to text inputs type

        :return: a text input type object
        """
        return self.keras_io.copy()
