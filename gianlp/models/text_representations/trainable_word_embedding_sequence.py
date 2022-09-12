"""
Module for pre-trained word embedding sequence input
"""

import pickle
import random
from collections import Counter
from typing import List, Optional, Callable, Union, Dict, cast

import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors

# pylint: disable=no-name-in-module
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Add, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence as keras_seq

# pylint: enable=no-name-in-module
from gianlp.keras_layers.masked_embedding import MaskedEmbedding
from gianlp.models.base_model import ModelIOShape
from gianlp.models.text_representations.text_representation import TextRepresentation
from gianlp.types import KerasInputOutput, SimpleTypeTexts


class TrainableWordEmbeddingSequence(TextRepresentation):
    """
    Trainable word embedding sequence input

    :var _keras_model: Keras model built from processing the text input
    :var _word2vec: the gensim keyedvectors object that contains the word embedding matrix
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
    _word2vec: Optional[KeyedVectors]
    _tokenizer: Callable[[str], List[str]]
    _pretrained_trainable: bool
    _sequence_maxlen: int
    _embedding_dimension: int
    _min_freq_percentile: float
    _max_vocabulary: Optional[int]
    _word_indexes: Dict[str, int]
    _random_state: int

    _WORD_UNKNOWN_TOKEN = "<UNK>"
    _MAX_SAMPLE_TO_FIT = 5000000

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        embedding_dimension: int,
        word2vec_src: Optional[Union[str, KeyedVectors]] = None,
        sequence_maxlen: int = 20,
        min_freq_percentile: float = 5,
        max_vocabulary: Optional[int] = None,
        pretrained_trainable: bool = False,
        random_state: int = 42,
    ):
        """
        :param tokenizer: A tokenizer function that transforms each string into a list of string tokens.
            The tokens transformed should match the keywords in the pretrained word embeddings.
            The function must support serialization through pickle
        :param word2vec_src: optional path to word2vec format .txt file or gensim KeyedVectors.
            if provided the common words from the corpus that are in the embedding will have this vectors assigned
        :param min_freq_percentile: the minimum percentile of the frequency to consider a word part of the vocabulary
        :param max_vocabulary: optional maximum vocabulary size
        :param embedding_dimension: the dimension of the target embedding
        :param sequence_maxlen: The maximum allowed sequence length
        :param pretrained_trainable: if the vectors pretrained will also be trained. ignored if word2vec_src is None
        :param random_state: the random seed used for random processes
        :raises ValueError: if a pretrained embeddings is fed and it's dimension does not match
            the one in embedding_dimension
        """
        super().__init__()
        if isinstance(word2vec_src, str):
            self._word2vec = KeyedVectors.load_word2vec_format(word2vec_src)
        else:
            self._word2vec = word2vec_src

        self._embedding_dimension = int(embedding_dimension)

        if self._word2vec:
            if self._word2vec.vector_size != self._embedding_dimension:
                raise ValueError(
                    f"The dimension of the pre-trained embeddings provided {self._word2vec.vector_size} "
                    f"does not match the target embedding dimension {self._embedding_dimension}."
                )

        self._tokenizer = tokenizer
        self._keras_model = None
        self._sequence_maxlen = int(sequence_maxlen)
        self._pretrained_trainable = pretrained_trainable
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

        tokenized_texts = self.tokenize_texts(texts, self._tokenizer, sequence_maxlength=self._sequence_maxlen)  # type: ignore[arg-type]
        words = keras_seq.pad_sequences(
            [
                [
                    self._word_indexes[w] if w in self._word_indexes else self._word_indexes[self._WORD_UNKNOWN_TOKEN]
                    for w in words
                ]
                for words in tokenized_texts
            ],
            maxlen=self._sequence_maxlen,
            dtype="int32",
            padding="post",
            truncating="post",
            value=0,
        )
        return cast(np.ndarray, words)

    def _unitary_build(self, texts: SimpleTypeTexts) -> None:
        """
        Builds the model using its inputs

        :param texts: the texts input
        """
        if not self._built:
            text_sample = texts.copy()
            random.seed(self._random_state)
            random.shuffle(text_sample)
            text_sample = text_sample[: min(len(text_sample), self._MAX_SAMPLE_TO_FIT)]
            tokenized_texts = self.tokenize_texts(
                text_sample, self._tokenizer, sequence_maxlength=self._sequence_maxlen  # type: ignore[arg-type]
            )
            frequencies = Counter([token for text in tokenized_texts for token in text])
            p_freq = np.percentile(list(frequencies.values()), self._min_freq_percentile)
            if (
                not self._max_vocabulary is None
                and (1 - (self._min_freq_percentile / 100)) * len(frequencies) > self._max_vocabulary
            ):
                vocabulary = [k for k, _ in frequencies.most_common(self._max_vocabulary)]
            else:
                vocabulary = [k for k, v in frequencies.items() if v >= p_freq]

            known_words = [k for k in vocabulary if self._word2vec and k in self._word2vec.key_to_index.keys()]
            new_words = [k for k in vocabulary if not self._word2vec or not k in self._word2vec.key_to_index.keys()]
            known_embeddings = np.zeros((len(known_words) + 1, self._embedding_dimension))
            np.random.seed(self._random_state)
            new_embeddings = np.concatenate(
                (
                    np.zeros((1, self._embedding_dimension)),
                    np.random.normal(0, 1, size=(len(new_words) + 1, self._embedding_dimension)),
                )
            )

            for i in range(len(known_words)):
                self._word2vec: KeyedVectors
                known_embeddings[i + 1, :] = self._word2vec.vectors[self._word2vec.key_to_index[known_words[i]]]
                self._word_indexes[known_words[i]] = i + 1

            self._word_indexes[self._WORD_UNKNOWN_TOKEN] = len(known_words) + 2

            for i in range(len(new_words)):
                self._word_indexes[new_words[i]] = len(known_words) + 3 + i

            inp = Input(shape=(self._sequence_maxlen,), dtype="int32")

            old_embeddings = MaskedEmbedding(
                input_dim=known_embeddings.shape[0],
                output_dim=known_embeddings.shape[1],
                weights=[known_embeddings],
                trainable=self._pretrained_trainable,
            )

            new_embeddings = MaskedEmbedding(
                input_dim=new_embeddings.shape[0],
                output_dim=new_embeddings.shape[1],
                weights=[new_embeddings],
                trainable=True,
            )

            new_index_words = Lambda(lambda x: K.relu(x - (len(known_words) + 1)))(inp)

            old_index_words = Lambda(lambda x: tf.cast(x == 0, "int32"))(new_index_words)
            old_index_words = Multiply()([old_index_words, inp])

            embedding = Add()([old_embeddings(old_index_words), new_embeddings(new_index_words)])

            self._keras_model = Model(inputs=inp, outputs=embedding)
            self._built = True

    @property
    def outputs_shape(self) -> ModelIOShape:
        """
        Returns the output shape of the model

        :return: a list of shape tuple or shape tuple
        """
        return ModelIOShape((self._sequence_maxlen, self._embedding_dimension))

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
                self._embedding_dimension,
                self._word2vec,
                self._sequence_maxlen,
                self._min_freq_percentile,
                self._max_vocabulary,
                self._pretrained_trainable,
                self._random_state,
            )
        )

    @classmethod
    def loads(cls, data: bytes) -> "TrainableWordEmbeddingSequence":
        """
        Loads a model

        :param data: the source bytes to load the model
        :return: a Serializable Model
        """
        (
            model_bytes,
            built,
            word_indexes,
            tokenizer,
            embedding_dimension,
            word2vec,
            sequence_maxlen,
            min_freq_percentile,
            max_vocabulary,
            pretrained_trainable,
            random_state,
        ) = pickle.loads(data)
        obj = cls(
            tokenizer,
            embedding_dimension,
            word2vec,
            sequence_maxlen,
            min_freq_percentile,
            max_vocabulary,
            pretrained_trainable,
            random_state,
        )
        if model_bytes:
            obj._keras_model = cls.get_model_from_bytes(model_bytes)
            obj._built = built
            obj._word_indexes = word_indexes
        return obj

    def _get_keras_model(self) -> Model:
        """
        Gets the internal keras model that is being serialized

        :return: The internal keras model
        """
        assert self._keras_model

        return self._keras_model
