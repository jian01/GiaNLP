"""
Module for fasttext embedding sequence input
"""

import pickle
from typing import List, Optional, Callable, Union, cast

import numpy as np
from functools import partial
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model, ft_ngram_hashes, FastTextKeyedVectors

# pylint: disable=no-name-in-module
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Embedding, Layer
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow import ragged
from tensorflow import int32

# pylint: enable=no-name-in-module

from gianlp.models.base_model import ModelIOShape
from gianlp.models.text_representations.text_representation import TextRepresentation
from gianlp.types import SimpleTypeTexts, KerasInputOutput


class FasttextNgramEmbeddingSequence(TextRepresentation):
    """
    Fasttext embedding sequence input

    :var _keras_model: Keras model built from processing the text input
    :var _fasttext: the gensim fasttext object
    :var _tokenizer: word tokenizer function
    :var _sequence_maxlen: the max length of an allowed sequence
    :var _trainable: if the pretrained vectors are trainable
    :var _ngram_indexes: the word to index dictionary
    :var _random_state: random seed
    :var min_n: the minimum ngram length
    :var max_n: the maximum ngram length
    :var bucket: the number of buckets
    :var _vector_size: the word vector size
    """

    _keras_model: Optional[Model]
    _fasttext: Optional[FastText]
    _max_n: int
    _min_n: int
    _bucket: int
    _tokenizer: Callable[[str], List[str]]
    _sequence_maxlen: int
    _normalized: bool
    _trainable: bool
    _random_state: int
    _vector_size: int

    PROPORTION_OF_MAXIMUM_BUCKETS = 1000

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        fasttext_src: Union[str, FastText],
        normalized: bool = False,
        trainable: bool = False,
        sequence_maxlen: int = 20,
        random_state: int = 42,
    ):
        """
        :param tokenizer: a tokenizer function that transforms each string into a list of string tokens
                            the tokens transformed should match the keywords in the pretrained word embeddings
                            the function must support serialization through pickle
        :param fasttext_src: path to fasttext facebook format .bit file or gensim FastText object.
        :param normalized: If True, the result vector of each word is normalized with euclidean length
        :param trainable: if the ngram embeddings are trainable, False is recommended.
        :param sequence_maxlen: The maximum allowed sequence length
        :param random_state: the random seed used for random processes
        """
        super().__init__()
        self._fasttext = None
        if fasttext_src:
            if isinstance(fasttext_src, str):
                self._fasttext = load_facebook_model(fasttext_src)
            else:
                self._fasttext = fasttext_src

        self._tokenizer = tokenizer
        self._max_n = self._fasttext.wv.max_n if self._fasttext else None
        self._min_n = self._fasttext.wv.min_n if self._fasttext else None
        self._bucket = self._fasttext.wv.bucket if self._fasttext else None
        self._vector_size = self._fasttext.vector_size if self._fasttext else None
        self._keras_model = None
        self._normalized = normalized
        self._trainable = trainable
        self._sequence_maxlen = int(sequence_maxlen)
        self._random_state = random_state

    @staticmethod
    def word_tokenizer_with_fasttext_ngrams(
        text: str, word_maxlen: int, tokenizer: Callable[[str], List[str]], min_n: int, max_n: int, bucket: int
    ) -> List[List[int]]:
        """
        Tokenizer wrapper for fasttext ngrams tokenizer

        :param text: the text to tokenize
        :param word_maxlen: the max length in word dimension
        :param tokenizer: the word tokenizer
        :param min_n: the minimum ngram length
        :param max_n: the maximum ngram length
        :param bucket: the number of buckets
        :return: a list of lists of ngrams ids per each word
        """
        text = tokenizer(text)
        tokenized = []
        for w in text:
            tokens = [
                t + 1
                for t in ft_ngram_hashes(w, min_n, max_n, bucket)[
                    : FasttextNgramEmbeddingSequence.PROPORTION_OF_MAXIMUM_BUCKETS * bucket
                ]
            ]
            if len(tokens) == 0:
                tokens.append(0)
            tokenized.append(tokens)
        if len(tokenized) < word_maxlen:
            tokenized += [[0]] * (word_maxlen - len(tokenized))
        return tokenized

    def preprocess_texts(self, texts: SimpleTypeTexts) -> KerasInputOutput:
        """
        Given texts returns the array representation needed for forwarding the
        keras model

        :param texts: the texts to preprocess
        :return: a numpy array of shape (#texts, #words, a dynamic amount of buckets)
        """
        assert self._tokenizer

        fasttext_tokenizer = partial(
            self.word_tokenizer_with_fasttext_ngrams,
            word_maxlen=self._sequence_maxlen,
            tokenizer=self._tokenizer,
            min_n=self._min_n,
            max_n=self._max_n,
            bucket=self._bucket,
        )

        tokenized_texts = self.tokenize_texts(texts, fasttext_tokenizer, sequence_maxlength=self._sequence_maxlen)  # type: ignore[arg-type]
        return ragged.constant(tokenized_texts)

    def _unitary_build(self, texts: SimpleTypeTexts) -> None:
        """
        Builds the model using its inputs

        :param texts: the texts input
        """

        if not self._built:
            self._fasttext = cast(FastText, self._fasttext)

            vectors = self._fasttext.wv.vectors_ngrams
            embeddings = np.concatenate((np.zeros((1, self._vector_size)), vectors))
            self._fasttext = None
            inp = Input(shape=(self._sequence_maxlen, None), dtype="int32", ragged=True)
            embedding = Embedding(
                input_dim=embeddings.shape[0],
                output_dim=embeddings.shape[1],
                weights=[embeddings],
                trainable=self._trainable,
            )(inp)
            add = Lambda(lambda x: K.sum(x, axis=-2))(embedding).to_tensor()
            if self._normalized:
                add = tf.linalg.l2_normalize(add, axis=-1)
            self._keras_model = Model(inputs=inp, outputs=add)
            self._built = True

    @property
    def inputs_shape(self) -> Union[List[ModelIOShape], ModelIOShape]:
        """
        Returns the shapes of the inputs of the model

        :return: a list of shape tuple or shape tuple
        """
        return ModelIOShape((self._sequence_maxlen, None), int32, ragged=True)

    @property
    def outputs_shape(self) -> ModelIOShape:
        """
        Returns the output shape of the model

        :return: a list of shape tuple or shape tuple
        """
        return ModelIOShape((self._sequence_maxlen, self._vector_size))

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
                self._tokenizer,
                self._fasttext,
                self._sequence_maxlen,
                self._min_n,
                self._max_n,
                self._bucket,
                self._normalized,
                self._trainable,
                self._random_state,
                self._vector_size,
            )
        )

    @classmethod
    def loads(cls, data: bytes) -> "FasttextNgramEmbeddingSequence":
        """
        Loads a model

        :param data: the source bytes to load the model
        :return: a Serializable Model
        """

        (
            model_bytes,
            built,
            tokenizer,
            fasttext,
            sequence_maxlen,
            min_n,
            max_n,
            bucket,
            normalized,
            trainable,
            random_state,
            vector_size,
        ) = pickle.loads(data)
        obj = cls(
            tokenizer,
            fasttext,
            normalized=normalized,
            trainable=trainable,
            sequence_maxlen=sequence_maxlen,
            random_state=random_state,
        )
        if model_bytes:
            obj._keras_model = cls.get_model_from_bytes(model_bytes)
            obj._built = built
            obj._min_n = min_n
            obj._max_n = max_n
            obj._bucket = bucket
            obj._vector_size = vector_size
        return obj

    def _get_keras_model(self) -> Model:
        """
        Gets the internal keras model that is being serialized

        :return: The internal keras model
        """
        assert self._keras_model

        return self._keras_model
