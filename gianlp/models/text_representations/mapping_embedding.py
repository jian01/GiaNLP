"""
Module for pre-trained embedding as a text-mapping
"""

import pickle
from typing import List, Optional, Union

import numpy as np
from gensim.models import KeyedVectors

# pylint: disable=no-name-in-module
from tensorflow import int32
from tensorflow.keras.layers import Input, Flatten, Embedding
from tensorflow.keras.models import Model

# pylint: enable=no-name-in-module

from gianlp.models.base_model import ModelIOShape
from gianlp.models.text_representations.text_representation import TextRepresentation
from gianlp.types import SimpleTypeTexts, KerasInputOutput


class MappingEmbedding(TextRepresentation):
    """
    Pre-trained word embedding sequence input

    :var _keras_model: Keras model built from processing the text input
    :var _keyed_vectors: the gensim keyedvectors object that contains the word embedding matrix
    """

    _keras_model: Optional[Model]
    _keyed_vectors: KeyedVectors

    _UNKNOWN_TOKEN = "<UNK>"

    def __init__(self, word2vec_src: Union[str, KeyedVectors]):
        """

        :param word2vec_src: path to word2vec format .txt file or gensim KeyedVectors
        """
        super().__init__()
        if isinstance(word2vec_src, str):
            self._keyed_vectors = KeyedVectors.load_word2vec_format(word2vec_src)
        else:
            self._keyed_vectors = word2vec_src
        self._keras_model = None

    def preprocess_texts(self, texts: SimpleTypeTexts) -> KerasInputOutput:
        """
        Given texts returns the array representation needed for forwarding the
        keras model

        :param texts: the texts to preprocess
        :return: a numpy array of shape (#texts, _sequence_maxlen)
        """
        assert self._keyed_vectors

        words = [
            [self._keyed_vectors.key_to_index[t] + 1 if t in self._keyed_vectors.key_to_index else 0] for t in texts
        ]
        return np.asarray(words)

    def _unitary_build(self, texts: SimpleTypeTexts) -> None:
        """
        Builds the model using its inputs

        :param texts: a text list for building if needed
        """
        if not self._built:
            embeddings = np.concatenate(
                (
                    np.mean(self._keyed_vectors.vectors, axis=0, keepdims=True),
                    np.random.normal(
                        0, 1, size=(len(self._keyed_vectors.key_to_index), self._keyed_vectors.vector_size)
                    ),
                )
            )
            embeddings[1:] = self._keyed_vectors.vectors
            inp = Input(shape=(1,), dtype="int32")
            embedding = Embedding(
                input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings], trainable=False
            )(inp)
            flattened = Flatten()(embedding)
            self._keras_model = Model(inputs=inp, outputs=flattened)
            self._built = True

    @property
    def inputs_shape(self) -> Union[List[ModelIOShape], ModelIOShape]:
        """
        Returns the shapes of the inputs of the model

        :return: a list of shape tuple or shape tuple
        """
        return ModelIOShape((1,), int32)

    @property
    def outputs_shape(self) -> ModelIOShape:
        """
        Returns the output shape of the model

        :return: a list of shape tuple or shape tuple
        """
        return ModelIOShape((self._keyed_vectors.vector_size,))

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
                self._keyed_vectors,
                self._built,
            )
        )

    @classmethod
    def loads(cls, data: bytes) -> "MappingEmbedding":
        """
        Loads a model

        :param data: the source bytes to load the model
        :return: a Serializable Model
        """
        model_bytes, keyed_vectors, built = pickle.loads(data)
        obj = cls(keyed_vectors)
        if model_bytes:
            obj._keras_model = cls.get_model_from_bytes(model_bytes)
            obj._built = built
        return obj

    def _get_keras_model(self) -> Model:
        """
        Get's the internal keras model that is being serialized

        :return: The internal keras model
        """
        assert self._keras_model

        return self._keras_model
