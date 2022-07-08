"""
Module for char sequence per word input
"""

import pickle
from typing import List, Callable, Optional, Union

import numpy as np

# pylint: disable=no-name-in-module
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
# pylint: enable=no-name-in-module

from gianlp.models.base_model import BaseModel
from gianlp.models.base_model import SimpleTypeTexts, ModelIOShape
from gianlp.models.text_representations.text_representation import TextRepresentation
from gianlp.models.trainable_model import KerasInputOutput


class PerChunkSequencer(TextRepresentation):
    """
    Per chunk sequencer wrapper
    For each chunk creates a sequence using the text input provided

    :var _chunker: function used for chunking the texts
    :var _sequencer: text input use for sequencing each chunk
    :var _chunking_maxlen: the maximum length in chunks for a text
    """

    _keras_model: Optional[Model]
    _chunker: Callable[[str], List[str]]
    _sequencer: TextRepresentation
    _chunking_maxlen: int

    def __init__(self, sequencer: TextRepresentation, chunker: Callable[[str], List[str]], chunking_maxlen: int):
        """

        :param sequencer: TextInput object
        :param chunker: function for chunking texts
        :param chunking_maxlen: the maximum length in chunks for a text
        """
        super().__init__()
        self._keras_model = None
        self._chunker = chunker
        self._sequencer = sequencer
        self._chunking_maxlen = int(chunking_maxlen)

    def preprocess_texts(self, texts: SimpleTypeTexts) -> KerasInputOutput:
        """
        Given texts returns the array representation needed for forwarding the
        keras model

        :param texts: the texts to preprocess
        :return: a numpy array of shape (#texts, _sequence_maxlen)
        """
        tokenized = [self._chunker(text) for text in texts]

        tokenized = [
            chunks[: self._chunking_maxlen]
            if len(chunks) >= self._chunking_maxlen
            else chunks + [""] * (self._chunking_maxlen - len(chunks))
            for chunks in tokenized
        ]

        tokenized = [self._sequencer.preprocess_texts(chunks) for chunks in tokenized]
        return np.asarray(tokenized)

    def _unitary_build(self, texts: SimpleTypeTexts) -> None:
        """
        Builds the model using its inputs

        :param texts: a text list for building if needed
        """
        if not self._built:
            inp = Input(shape=self.outputs_shape.shape[:-1], dtype=self.outputs_shape.dtype)
            out = self._sequencer(inp)
            self._keras_model = Model(inputs=inp, outputs=out)
            self._built = True

    def build(self, texts: SimpleTypeTexts) -> None:
        """
        Builds the whole chain of models in a recursive manner using the functional API

        :param texts: the texts for building if needed
        """
        self._sequencer.build(texts)
        super().build(texts)

    @property
    def outputs_shape(self) -> Union[List[ModelIOShape], ModelIOShape]:
        """
        Returns the output shape of the model

        :return: a list of shape tuple or shape tuple
        """
        return ModelIOShape((self._chunking_maxlen,) + self._sequencer.outputs_shape.shape,
                            self._sequencer.outputs_shape.dtype)

    def dumps(self) -> bytes:
        """
        Dumps the model into bytes

        :return: a byte array
        """
        model_bytes = None
        if self._keras_model:
            model_bytes = self.get_bytes_from_model(self._keras_model)
        return pickle.dumps(
            (
                model_bytes,
                self._sequencer.serialize(),
                self._chunker,
                self._chunking_maxlen,
                self._built,
            )
        )

    @classmethod
    def loads(cls, data: bytes) -> "PerChunkSequencer":
        """
        Loads a model

        :param data: the source bytes to load the model
        :return: a Serializable Model
        """
        model_bytes, sequencer_bytes, chunker, chunking_maxlen, _built = pickle.loads(data)
        obj = cls(BaseModel.deserialize(sequencer_bytes), chunker, chunking_maxlen)
        if model_bytes:
            obj._keras_model = cls.get_model_from_bytes(model_bytes)
            obj._built = _built
        return obj

    def _get_keras_model(self) -> Model:
        """
        Get's the internal keras model that is being serialized

        :return: The internal keras model
        """
        assert self._keras_model

        return self._keras_model
