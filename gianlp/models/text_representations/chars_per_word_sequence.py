"""
Module for char sequence per word input
"""

import pickle
import random
from collections import Counter
from typing import List, Optional, Dict, Callable, Union

import numpy as np

# pylint: disable=no-name-in-module
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence as keras_seq
# pylint: enable=no-name-in-module

from gianlp.keras_layers.masked_embedding import MaskedEmbedding
from gianlp.models.base_model import SimpleTypeTexts, ModelIOShape
from gianlp.models.text_representations.text_representation import TextRepresentation
from gianlp.models.trainable_model import KerasInputOutput


class CharPerWordEmbeddingSequence(TextRepresentation):
    """
    Char per word sequence

    :var _char_indexes: mapping from char to index
    :var _keras_model: Keras model built from processing the text input
    :var _tokenizer: word tokenizer function
    :var _embedding_dimension: the dimension of the embedding
    :var _word_maxlen: the max length for word sequences
    :var _char_maxlen: the max length for chars within a word
    :var _min_freq_percentile: the minimum frequency percentile to consider a char as known
    :var _random_state: the random seed used for randomized operations
    """

    _char_indexes: Optional[Dict[str, int]]
    _keras_model: Optional[Model]
    _tokenizer: Callable[[str], List[str]]
    _embedding_dimension: int
    _word_maxlen: int
    _char_maxlen: int
    _min_freq_percentile: int
    _random_state: int

    MAX_SAMPLE_TO_FIT = 1000000
    CHAR_EMB_UNK_TOKEN = "UNK"

    def __init__(
            self,
            tokenizer: Callable[[str], List[str]],
            embedding_dimension: int = 256,
            word_maxlen: int = 30,
            char_maxlen: int = 12,
            min_freq_percentile: int = 5,
            random_state: int = 42,
    ):
        """

        :param tokenizer: a tokenizer function that transforms each string into a list of string tokens
                        the function must support serialization through pickle
        :param embedding_dimension: The char embedding dimension
        :param word_maxlen: the max length for word sequences
        :param char_maxlen: the max length for chars within a word
        :param min_freq_percentile: minimum percentile of the frequency for keeping a char.
                                    If a char has a frequency lower than this percentile it
                                    would be treated as unknown.
        :param random_state: random seed
        """
        super().__init__()
        self._char_indexes = None
        self._keras_model = None
        self._tokenizer = tokenizer
        self._embedding_dimension = int(embedding_dimension)
        self._word_maxlen = int(word_maxlen)
        self._char_maxlen = int(char_maxlen)
        self._min_freq_percentile = min_freq_percentile
        self._random_state = random_state

    def preprocess_texts(self, texts: SimpleTypeTexts) -> KerasInputOutput:
        """
        Given texts returns the array representation needed for forwarding the
        keras model

        :param texts: the texts to preprocess
        :return: a numpy array of shape (#texts, _sequence_maxlen)
        """
        assert self._char_indexes

        tokenized = [self._tokenizer(text) for text in texts]
        tokenized = [
            [[self._char_indexes[c] if c in self._char_indexes else self._char_indexes[self.CHAR_EMB_UNK_TOKEN] for c in
              w] for w in text]
            for text in tokenized
        ]

        tokenized = [
            text[: self._word_maxlen] if len(text) >= self._word_maxlen else text + [[]] * (
                    self._word_maxlen - len(text))
            for text in tokenized
        ]
        tokenized = [keras_seq.pad_sequences(text, maxlen=self._char_maxlen, padding="post", truncating="post") for text
                     in tokenized]
        return np.asarray(tokenized)

    def _unitary_build(self, texts: SimpleTypeTexts) -> None:
        """
        Builds the model using its inputs

        :param texts: a text list for building if needed
        """
        if not self._built:
            text_sample = texts.copy()
            random.seed(self._random_state)
            random.shuffle(text_sample)
            text_sample = text_sample[: min(len(text_sample), self.MAX_SAMPLE_TO_FIT)]

            char_ocurrences = [list(text) for text in text_sample]
            char_ocurrence_counter = Counter()  # type: ignore
            for seq in char_ocurrences:
                char_ocurrence_counter.update(seq)
            p_freq = np.percentile(list(char_ocurrence_counter.values()), self._min_freq_percentile)
            char_ocurrence_dict = {k: v for k, v in char_ocurrence_counter.items() if v > p_freq}
            self._char_indexes = {
                count[0]: i + 1 for i, count in
                enumerate(Counter(char_ocurrence_dict).most_common(len(char_ocurrence_dict)))
            }
            self._char_indexes[self.CHAR_EMB_UNK_TOKEN] = len(self._char_indexes) + 1
            self.__init_keras_model()
            self._built = True

    def __init_keras_model(self) -> None:
        """
        Creates the keras model ready to represent the output of the text
        preprocessor

        :return: a keras Model
        """
        assert self._char_indexes
        if not self._keras_model:
            np.random.seed(self._random_state)
            embedding_init = np.random.normal(size=(len(self._char_indexes), self._embedding_dimension))
            embedding_init = np.vstack([np.zeros((1, self._embedding_dimension)), embedding_init])
            inp = Input(shape=(self._word_maxlen, self._char_maxlen), dtype="int32")
            embedding = MaskedEmbedding(
                input_dim=len(self._char_indexes) + 1,
                output_dim=self._embedding_dimension,
                trainable=True,
                weights=[embedding_init],
            )(inp)
            self._keras_model = Model(inputs=inp, outputs=embedding)

    @property
    def outputs_shape(self) -> Union[List[ModelIOShape], ModelIOShape]:
        """
        Returns the output shape of the model

        :return: a list of shape tuple or shape tuple
        """
        return ModelIOShape((self._word_maxlen, self._char_maxlen, self._embedding_dimension))

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
                self._char_indexes,
                model_bytes,
                self._tokenizer,
                self._embedding_dimension,
                self._word_maxlen,
                self._char_maxlen,
                self._min_freq_percentile,
                self._random_state,
                self._built,
            )
        )

    @classmethod
    def loads(cls, data: bytes) -> "CharPerWordEmbeddingSequence":
        """
        Loads a model

        :return: a Serializable Model
        """
        (
            _char_indexes,
            model_bytes,
            tokenizer,
            embedding_dimension,
            word_maxlen,
            char_maxlen,
            min_freq_percentile,
            random_state,
            _built,
        ) = pickle.loads(data)
        obj = cls(tokenizer, embedding_dimension, word_maxlen, char_maxlen, min_freq_percentile, random_state)
        obj._char_indexes = _char_indexes
        if model_bytes:
            obj._keras_model = cls.get_model_from_bytes(model_bytes)
            obj._built = _built
        return obj

    def _get_keras_model(self) -> Model:
        """
        Get's the internal keras model that is being serialized
        """
        assert self._keras_model

        return self._keras_model
