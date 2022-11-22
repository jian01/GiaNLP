"""
Char embedding sequence utils
"""

import unittest

import numpy as np
from tensorflow.keras.layers import Input

from gianlp.models import BaseModel, CharEmbeddingSequence
from tests.utils import LOREM_IPSUM


class TestCharEmbeddingSequence(unittest.TestCase):
    """
    Char embedding sequence utils
    """

    @staticmethod
    def create_char_emb(
            embedding_dimension: int = 16, sequence_maxlen: int = 10, random_state: int = 42
    ) -> CharEmbeddingSequence:
        """
        Creates a char embedding
        """
        char_emb = CharEmbeddingSequence(
            embedding_dimension=embedding_dimension, sequence_maxlen=sequence_maxlen, random_state=random_state
        )
        char_emb.build(LOREM_IPSUM.split(" "))
        return char_emb

    def test_preprocess(self) -> None:
        """
        Test char embedding sequence preprocess method
        """
        char_emb = self.create_char_emb()
        preprocessed1 = char_emb.preprocess_texts(["prueba"])
        preprocessed2 = char_emb.preprocess_texts(["prueba"])
        self.assertEqual(preprocessed1.tolist(), preprocessed2.tolist())

    def test_weights(self) -> None:
        """
        Test weights count
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)
        self.assertIsNone(char_emb.weights_amount)
        self.assertIsNone(char_emb.trainable_weights_amount)
        char_emb.build(LOREM_IPSUM.split(" "))
        self.assertEqual(char_emb.weights_amount, char_emb.trainable_weights_amount)

    def test_maxlen(self) -> None:
        """
        Test maxlen of a sequence
        """
        char_emb = self.create_char_emb(sequence_maxlen=10)
        preprocessed = char_emb.preprocess_texts(["prueba"])
        self.assertTrue(preprocessed[:6].sum() > 6)
        self.assertTrue(not preprocessed[6:].any())

    def test_serialize(self) -> None:
        """
        Test object serialization
        """
        char_emb: CharEmbeddingSequence = self.create_char_emb()
        preprocessed1 = char_emb.preprocess_texts(["prueba"])
        data = char_emb.serialize()
        char_emb = BaseModel.deserialize(data)
        preprocessed2 = char_emb.preprocess_texts(["prueba"])
        self.assertEqual(preprocessed1.tolist(), preprocessed2.tolist())

    def test_functional_api_simple_layer(self) -> None:
        """
        Test object functional api
        """
        char_emb = self.create_char_emb(embedding_dimension=16, sequence_maxlen=10)
        inputs = Input(shape=(10,))
        outputs = char_emb(inputs)
        self.assertEqual(outputs.shape.as_list(), [None, 10, 16])

    def test_functional_api_multiple_inputs(self) -> None:
        """
        Test object functional api with multiple inputs
        """
        char_emb = self.create_char_emb(embedding_dimension=16, sequence_maxlen=10)
        inputs1 = Input(shape=(10,))
        inputs2 = Input(shape=(10,))
        outputs = char_emb([inputs1, inputs2])
        self.assertEqual(outputs.shape.as_list(), [None, 10, 32])

    def test_functional_api_multiple_sequence(self) -> None:
        """
        Test object functional api with sequence with more dims
        """
        char_emb = self.create_char_emb(embedding_dimension=16, sequence_maxlen=10)
        inputs = Input(shape=(100, 10))
        outputs = char_emb(inputs)
        self.assertEqual(outputs.shape.as_list(), [None, 100, 10, 16])
        inputs = Input(shape=(100, 12, 10))
        outputs = char_emb(inputs)
        self.assertEqual(outputs.shape.as_list(), [None, 100, 12, 10, 16])

    def test_functional_api_multi_sequence_inputs(self) -> None:
        """
        Test object functional api complex scenario
        """
        char_emb = self.create_char_emb(embedding_dimension=16, sequence_maxlen=10)
        inputs1 = Input(shape=(100, 10))
        inputs2 = Input(shape=(100, 10))
        outputs = char_emb([inputs1, inputs2])
        self.assertEqual(outputs.shape.as_list(), [None, 100, 10, 32])

    def test_functional_api_numpy_sequence(self) -> None:
        """
        Test object functional api
        """
        char_emb = self.create_char_emb(embedding_dimension=16, sequence_maxlen=10)
        outputs = char_emb(np.asarray([1, 2, 1, 2, 1, 0, 0, 0, 0, 0]).reshape((1, 10)))
        self.assertEqual(outputs.shape, (1, 10, 16))

    def test_random_seed(self) -> None:
        """
        Test object functional api
        """
        char_emb = self.create_char_emb(embedding_dimension=16, sequence_maxlen=10, random_state=42)
        char_emb2 = self.create_char_emb(embedding_dimension=16, sequence_maxlen=10, random_state=42)
        char_emb_diff_seed = self.create_char_emb(embedding_dimension=16, sequence_maxlen=10, random_state=2)
        outputs1 = char_emb(np.asarray([1, 2, 1, 2, 1, 0, 0, 0, 0, 0]).reshape((10,)))
        outputs2 = char_emb2(np.asarray([1, 2, 1, 2, 1, 0, 0, 0, 0, 0]).reshape((10,)))
        outputs3 = char_emb_diff_seed(np.asarray([1, 2, 1, 2, 1, 0, 0, 0, 0, 0]).reshape((10,)))

        self.assertEqual(outputs1.tolist(), outputs2.tolist())
        self.assertNotEqual(outputs1.tolist(), outputs3.tolist())
