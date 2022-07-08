"""
Module for pre-trained word embedding sequence input
"""

import pickle
from typing import List, Optional, Callable, Union

import numpy as np
from gensim.models import KeyedVectors

# pylint: disable=no-name-in-module
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence as keras_seq

# pylint: enable=no-name-in-module

from gianlp.keras_layers.masked_embedding import MaskedEmbedding
from gianlp.models.base_model import SimpleTypeTexts, ModelIOShape
from gianlp.models.text_representations.text_representation import TextRepresentation
from gianlp.models.trainable_model import KerasInputOutput


class PreTrainedWordEmbeddingSequence(TextRepresentation):
    """
    Pre-trained word embedding sequence input

    :var _keras_model: Keras model built from processing the text input
    :var _word2vec: the gensim keyedvectors object that contains the word embedding matrix
    :var _tokenizer: word tokenizer function
    :var _sequence_maxlen: the max length of an allowed sequence
    """

    _keras_model: Optional[Model]
    _word2vec: KeyedVectors
    _tokenizer: Callable[[str], List[str]]
    _sequence_maxlen: int

    WORD_UNKNOWN_TOKEN = "<UNK>"

    def __init__(
        self,
        word2vec_src: Union[str, KeyedVectors],
        tokenizer: Callable[[str], List[str]],
        sequence_maxlen: int = 20,
    ):
        """

        :param word2vec_src: path to word2vec format .txt file or gensim KeyedVectors
        :param tokenizer: a tokenizer function that transforms each string into a list of string tokens
                            the tokens transformed should match the keywords in the pretrained word embeddings
                            the function must support serialization through pickle
        :param sequence_maxlen: The maximum allowed sequence length"""
        super().__init__()
        if isinstance(word2vec_src, str):
            self._word2vec = KeyedVectors.load_word2vec_format(word2vec_src)
        else:
            self._word2vec = word2vec_src
        self._tokenizer = tokenizer
        self._keras_model = None
        self._sequence_maxlen = int(sequence_maxlen)

    def preprocess_texts(self, texts: SimpleTypeTexts) -> KerasInputOutput:
        """
        Given texts returns the array representation needed for forwarding the
        keras model

        :param texts: the texts to preprocess
        :return: a numpy array of shape (#texts, _sequence_maxlen)
        """
        assert self._tokenizer
        assert self._word2vec

        tokenized_texts = [self._tokenizer(text) for text in texts]
        words = keras_seq.pad_sequences(
            [
                [
                    self._word2vec.key_to_index[w] + 2 if w in self._word2vec.key_to_index else 1
                    for w in words[: self._sequence_maxlen]
                ]
                for words in tokenized_texts
            ],
            maxlen=self._sequence_maxlen,
            dtype="int32",
            padding="post",
            truncating="post",
            value=0,
        )
        return words

    def _unitary_build(self, texts: SimpleTypeTexts) -> None:
        """
        Builds the model using its inputs

        :param texts: a text list for building if needed
        """
        if not self._built:
            embeddings = np.concatenate(
                (
                    np.zeros((1, self._word2vec.vector_size)),
                    np.mean(self._word2vec.vectors, axis=0, keepdims=True),
                    np.random.normal(0, 1, size=(len(self._word2vec.key_to_index), self._word2vec.vector_size)),
                )
            )
            embeddings[2:] = self._word2vec.vectors
            inp = Input(shape=(self._sequence_maxlen,), dtype="int32")
            embedding = MaskedEmbedding(
                input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings], trainable=False
            )(inp)
            self._keras_model = Model(inputs=inp, outputs=embedding)
            self._built = True

    @property
    def outputs_shape(self) -> Union[List[ModelIOShape], ModelIOShape]:
        """
        Returns the output shape of the model

        :return: a list of shape tuple or shape tuple
        """
        return ModelIOShape((self._sequence_maxlen, self._word2vec.vector_size))

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
                self._word2vec,
                self._tokenizer,
                self._sequence_maxlen,
                self._built,
            )
        )

    @classmethod
    def loads(cls, data: bytes) -> "PreTrainedWordEmbeddingSequence":
        """
        Loads a model

        :param data: the source bytes to load the model
        :return: a Serializable Model
        """
        model_bytes, word2vec, tokenizer, sequence_maxlen, built = pickle.loads(data)
        obj = cls(word2vec, tokenizer, sequence_maxlen)
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
