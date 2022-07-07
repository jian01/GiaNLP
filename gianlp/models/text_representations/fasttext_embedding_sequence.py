"""
Module for pre-trained word embedding sequence input
"""

import pickle
import random
from collections import Counter
from typing import List, Optional, Callable, Union, Dict

import numpy as np
import tensorflow as tf
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model

# pylint: disable=no-name-in-module
from tensorflow.keras.layers import Input, Lambda, Add, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence as keras_seq
# pylint: enable=no-name-in-module

from gianlp.keras_layers.masked_embedding import MaskedEmbedding
from gianlp.models.base_model import SimpleTypeTexts, ModelIOShape
from gianlp.models.text_representations.text_representation import TextRepresentation
from gianlp.models.trainable_model import KerasInputOutput


class FasttextEmbeddingSequence(TextRepresentation):
    """
    Fasttext (non-trainable) embedding sequence input

    :var _keras_model: Keras model built from processing the text input
    :var _fasttext: the gensim fasttext object
    :var _tokenizer: word tokenizer function
    :var _pretrained_trainable: if the pretrained vectors are trainable
    :var _sequence_maxlen: the max length of an allowed sequence
    :var _embedding_dimension: target embedding dimension
    :var _min_freq_percentile: the minimum percentile of the frequency to consider a word part of the vocabulary
    :var _max_vocabulary: optional maximum vocabulary size
    :var _word_indexes: the word to index dictionary
    :var _random_state: random seed
    """

    _keras_model: Optional[Model]
    _fasttext: Optional[FastText]
    _tokenizer: Callable[[str], List[str]]
    _sequence_maxlen: int
    _min_freq_percentile: float
    _max_vocabulary: Optional[int]
    _word_indexes: Dict[str, int]
    _random_state: int

    WORD_UNKNOWN_TOKEN = "<UNK>"
    MAX_SAMPLE_TO_FIT = 5000000

    def __init__(
            self,
            tokenizer: Callable[[str], List[str]],
            fasttext_src: Optional[Union[str, FastText]] = None,
            sequence_maxlen: int = 20,
            min_freq_percentile: float = 5,
            max_vocabulary: Optional[int] = None,
            random_state: int = 42,
    ):
        """
        :param tokenizer: a tokenizer function that transforms each string into a list of string tokens
                            the tokens transformed should match the keywords in the pretrained word embeddings
                            the function must support serialization through pickle
        :param fasttext_src: optional path to fasttext facebook format .bit file or gensim FastText object.
        if provided the common words from the corpus that are in the embedding will have this vectors assigned
        :param min_freq_percentile: the minimum percentile of the frequency to consider a word part of the vocabulary
        :param max_vocabulary: optional maximum vocabulary size
        :param sequence_maxlen: The maximum allowed sequence length
        :param random_state: the random seed used for random processes
        """
        super().__init__()
        if isinstance(fasttext_src, str):
            self._fasttext = load_facebook_model(fasttext_src)
        else:
            self._fasttext = fasttext_src

        self._tokenizer = tokenizer
        self._keras_model = None
        self._sequence_maxlen = int(sequence_maxlen)
        self._min_freq_percentile = min_freq_percentile
        self._max_vocabulary = max_vocabulary
        self._random_state = random_state
        self._word_indexes = {}

    def preprocess_texts(self, texts: SimpleTypeTexts) -> KerasInputOutput:
        """
        Given texts returns the array representation needed for forwarding the
        keras model

        :param texts: the texts to preprocess
        :return: a numpy array of shape (#texts, _sequence_maxlen)
        """
        assert self._built

        tokenized_texts = [self._tokenizer(text) for text in texts]
        words = keras_seq.pad_sequences(
            [
                [
                    self._word_indexes[w] if w in self._word_indexes else self._word_indexes[self.WORD_UNKNOWN_TOKEN]
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
            text_sample = texts.copy()
            random.seed(self._random_state)
            random.shuffle(text_sample)
            text_sample = text_sample[: min(len(text_sample), self.MAX_SAMPLE_TO_FIT)]
            tokenized_texts = [token for text in text_sample for token in self._tokenizer(text)]
            frequencies = Counter(tokenized_texts)
            p_freq = np.percentile(list(frequencies.values()), self._min_freq_percentile)
            if not self._max_vocabulary is None and (1 - (self._min_freq_percentile / 100)) * len(
                    frequencies) > self._max_vocabulary:
                frequencies = frequencies.most_common(self._max_vocabulary)
            else:
                frequencies = [(k, v) for k, v in frequencies.items() if v >= p_freq]

            vocabulary = known_words = [k for k, _ in frequencies]

            auxiliar_matrix = np.concatenate(
                (
                    np.zeros((1, self._fasttext.vector_size)),
                    np.random.normal(0, 1, size=(1, self._fasttext.vector_size)),
                )
            )

            np.random.seed(self._random_state)
            emb_matrix = np.concatenate(
                (
                    np.zeros((2, self._fasttext.vector_size)),
                    np.random.normal(0, 1, size=(len(vocabulary) + 1, self._fasttext.vector_size)),
                )
            )

            self._word_indexes[self.WORD_UNKNOWN_TOKEN] = 1

            for i in range(len(vocabulary)):
                emb_matrix[i + 2, :] = self._fasttext.wv[vocabulary[i]]
                self._word_indexes[known_words[i]] = i + 2

            inp = Input(shape=(self._sequence_maxlen,), dtype="int32")

            auxiliar_embs = MaskedEmbedding(
                input_dim=auxiliar_matrix.shape[0], output_dim=auxiliar_matrix.shape[1], weights=[auxiliar_matrix],
                trainable=True
            )

            embeddings = MaskedEmbedding(
                input_dim=emb_matrix.shape[0], output_dim=emb_matrix.shape[1], weights=[emb_matrix], trainable=False
            )

            auxiliary_index = Lambda(lambda x: tf.cast(x <= 1, "int32"))(inp)
            auxiliary_index = Multiply()([auxiliary_index, inp])

            embedding = Add()([auxiliar_embs(auxiliary_index), embeddings(inp)])

            self._keras_model = Model(inputs=inp, outputs=embedding)
            self._built = True
            self._fasttext = FastText(size=self._fasttext.vector_size)

    @property
    def outputs_shape(self) -> Union[List[ModelIOShape], ModelIOShape]:
        """
        Returns the output shape of the model

        :return: a list of shape tuple or shape tuple
        """
        return ModelIOShape((self._sequence_maxlen, self._fasttext.vector_size))

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
                self._built,
                self._word_indexes,
                self._tokenizer,
                self._fasttext,
                self._sequence_maxlen,
                self._min_freq_percentile,
                self._max_vocabulary,
                self._random_state,
            )
        )

    @classmethod
    def loads(cls, data: bytes) -> "FasttextEmbeddingSequence":
        """
        Loads a model

        :return: a Serializable Model
        """
        (
            model_bytes,
            built,
            word_indexes,
            tokenizer,
            word2vec,
            sequence_maxlen,
            min_freq_percentile,
            max_vocabulary,
            random_state,
        ) = pickle.loads(data)
        obj = cls(tokenizer, word2vec, sequence_maxlen, min_freq_percentile, max_vocabulary, random_state)
        if model_bytes:
            obj._keras_model = cls.get_model_from_bytes(model_bytes)
            obj._built = built
            obj._word_indexes = word_indexes
        return obj

    def _get_keras_model(self) -> Model:
        """
        Get's the internal keras model that is being serialized
        """
        assert self._keras_model

        return self._keras_model
