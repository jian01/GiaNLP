"""
Pre trained word embedding sequence tests
"""

import unittest

import numpy as np
from tensorflow.keras.layers import Input

from gianlp.models import BaseModel
from gianlp.models import PreTrainedWordEmbeddingSequence
from tests.utils import split_tokenizer


class TestPreTrainedWordEmbeddingSequence(unittest.TestCase):
    """
    Pre trained word embedding sequence tests
    """

    @staticmethod
    def create_word_emb() -> PreTrainedWordEmbeddingSequence:
        """
        Creates a test word embedding
        """
        word_emb = PreTrainedWordEmbeddingSequence(
            "tests/resources/test_word2vec.txt", tokenizer=split_tokenizer, sequence_maxlen=4
        )
        word_emb.build([""])
        return word_emb

    def test_output_shape(self) -> None:
        """
        Test shape
        """
        word_emb = self.create_word_emb()
        self.assertEqual(word_emb.outputs_shape.shape, (4, 3))

    def test_preprocess(self) -> None:
        """
        Test preprocess method
        """
        word_emb = self.create_word_emb()
        preprocessed1 = word_emb.preprocess_texts(["hola de"])
        preprocessed2 = word_emb.preprocess_texts(["hola de"])
        self.assertEqual(preprocessed1.tolist(), preprocessed2.tolist())

    def test_weights(self) -> None:
        """
        Test weights count
        """
        char_emb = PreTrainedWordEmbeddingSequence("tests/resources/test_word2vec.txt", tokenizer=split_tokenizer, sequence_maxlen=4)
        self.assertIsNone(char_emb.weights_amount)
        self.assertIsNone(char_emb.trainable_weights_amount)
        char_emb.build([""])
        self.assertEqual(char_emb.weights_amount, 3 * 4 + 3 + 3)
        self.assertEqual(char_emb.trainable_weights_amount, 0)

    def test_maxlen(self) -> None:
        """
        Test maxlen of a sequence
        """
        char_emb = self.create_word_emb()
        preprocessed = char_emb.preprocess_texts(["prueba hola prueba"])
        self.assertTrue((preprocessed > 0).sum() == 3)
        self.assertTrue(not preprocessed[3:].any())
        preprocessed = char_emb.preprocess_texts(["prueba prueba hola prueba prueba"])
        self.assertTrue((preprocessed > 0).sum() == 4)

    def test_serialize(self) -> None:
        """
        Test object serialization
        """
        word_emb: PreTrainedWordEmbeddingSequence = self.create_word_emb()
        preprocessed1 = word_emb.preprocess_texts(["hola"])
        data = word_emb.serialize()
        word_emb = BaseModel.deserialize(data)
        preprocessed2 = word_emb.preprocess_texts(["hola"])
        self.assertEqual(preprocessed1.tolist(), preprocessed2.tolist())

    def test_functional_api_simple_layer(self) -> None:
        """
        Test object functional api
        """
        word_emb = self.create_word_emb()
        inputs = Input(shape=(4,))
        outputs = word_emb(inputs)
        self.assertEqual(outputs.shape.as_list(), [None, 4, 3])

    def test_functional_api_multiple_inputs(self) -> None:
        """
        Test object functional api with multiple inputs
        """
        word_emb = self.create_word_emb()
        inputs1 = Input(shape=(4,))
        inputs2 = Input(shape=(4,))
        outputs = word_emb([inputs1, inputs2])
        self.assertEqual(outputs.shape.as_list(), [None, 4, 6])

    def test_functional_api_multiple_sequence(self) -> None:
        """
        Test object functional api with sequence with more dims
        """
        word_emb = self.create_word_emb()
        inputs = Input(shape=(100, 4))
        outputs = word_emb(inputs)
        self.assertEqual(outputs.shape.as_list(), [None, 100, 4, 3])
        inputs = Input(shape=(100, 12, 4))
        outputs = word_emb(inputs)
        self.assertEqual(outputs.shape.as_list(), [None, 100, 12, 4, 3])

    def test_functional_api_multi_sequence_inputs(self) -> None:
        """
        Test object functional api complex scenario
        """
        word_emb = self.create_word_emb()
        inputs1 = Input(shape=(100, 4))
        inputs2 = Input(shape=(100, 4))
        outputs = word_emb([inputs1, inputs2])
        self.assertEqual(outputs.shape.as_list(), [None, 100, 4, 6])

    def test_functional_api_numpy_sequence(self) -> None:
        """
        Test object functional api
        """
        word_emb = self.create_word_emb()
        outputs = word_emb(np.asarray([1, 2, 1, 0]).reshape((1, 4)))
        self.assertEqual(outputs.shape, (1, 4, 3))
