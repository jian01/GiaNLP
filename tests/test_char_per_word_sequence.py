"""
Char embedding sequence tests
"""

import unittest

import numpy as np
from tensorflow.keras.layers import Input

from gianlp.models import BaseModel
from gianlp.models import CharPerWordEmbeddingSequence
from tests.utils import split_tokenizer, LOREM_IPSUM


class TestCharPerWordEmbeddingSequence(unittest.TestCase):
    """
    Char per word embedding sequence tests
    """

    @staticmethod
    def create_test_obj(random_seed: int = 42) -> CharPerWordEmbeddingSequence:
        """
        Creates a test embedding
        """
        char_emb = CharPerWordEmbeddingSequence(tokenizer=split_tokenizer, word_maxlen=4, char_maxlen=5, random_state=random_seed)
        char_emb.build(LOREM_IPSUM.split("\n"))
        return char_emb

    def test_preprocess(self) -> None:
        """
        Test preprocess method
        """
        char_seq = self.create_test_obj()
        preprocessed1 = char_seq.preprocess_texts(["hola de"])
        preprocessed2 = char_seq.preprocess_texts(["hola de"])
        self.assertEqual(preprocessed1.tolist(), preprocessed2.tolist())

    def test_maxlen(self) -> None:
        """
        Test maxlen of a sequence
        """
        char_seq = self.create_test_obj()
        preprocessed = char_seq.preprocess_texts(["prueba hola prueba"])
        self.assertTrue((preprocessed > 0).sum() == 14)
        self.assertTrue(not preprocessed[3:].any())
        preprocessed = char_seq.preprocess_texts(["prueba prueba hola prueba prueba"])
        self.assertTrue((preprocessed > 0).sum() == 19)
        self.assertEqual(preprocessed.shape, (1, 4, 5))

    def test_serialize(self) -> None:
        """
        Test object serialization
        """
        char_seq: CharPerWordEmbeddingSequence = self.create_test_obj()
        preprocessed1 = char_seq.preprocess_texts(["hola"])
        data = char_seq.serialize()
        char_seq = BaseModel.deserialize(data)
        preprocessed2 = char_seq.preprocess_texts(["hola"])
        self.assertEqual(preprocessed1.tolist(), preprocessed2.tolist())

    def test_functional_api_simple_layer(self) -> None:
        """
        Test object functional api
        """
        char_seq = self.create_test_obj()
        inputs = Input(shape=(4, 5))
        outputs = char_seq(inputs)
        self.assertEqual(outputs.shape.as_list(), [None, 4, 5, 256])

    def test_functional_api_multiple_inputs(self) -> None:
        """
        Test object functional api with multiple inputs
        """
        char_seq = self.create_test_obj()
        inputs1 = Input(shape=(4, 5))
        inputs2 = Input(shape=(4, 5))
        outputs = char_seq([inputs1, inputs2])
        self.assertEqual(outputs.shape.as_list(), [None, 4, 5, 512])

    def test_functional_api_multiple_sequence(self) -> None:
        """
        Test object functional api with sequence with more dims
        """
        char_seq = self.create_test_obj()
        inputs = Input(shape=(100, 4, 5))
        outputs = char_seq(inputs)
        self.assertEqual(outputs.shape.as_list(), [None, 100, 4, 5, 256])
        inputs = Input(shape=(100, 12, 4, 5))
        outputs = char_seq(inputs)
        self.assertEqual(outputs.shape.as_list(), [None, 100, 12, 4, 5, 256])

    def test_functional_api_multi_sequence_inputs(self) -> None:
        """
        Test object functional api complex scenario
        """
        char_seq = self.create_test_obj()
        inputs1 = Input(shape=(100, 4, 5))
        inputs2 = Input(shape=(100, 4, 5))
        outputs = char_seq([inputs1, inputs2])
        self.assertEqual(outputs.shape.as_list(), [None, 100, 4, 5, 512])

    def test_functional_api_numpy_sequence(self) -> None:
        """
        Test object functional api
        """
        char_seq = self.create_test_obj()
        numpy_input = np.random.randint(0, 10, size=(1, 4, 5))
        outputs = char_seq(numpy_input)
        self.assertEqual(outputs.shape, (1, 4, 5, 256))

    def test_random_seed(self) -> None:
        """
        Test object functional api
        """
        char_seq = self.create_test_obj(random_seed=42)
        char_seq2 = self.create_test_obj(random_seed=42)
        char_seq_diff_seed = self.create_test_obj(random_seed=1)
        numpy_input = np.random.randint(0, 10, size=(1, 4, 5))
        outputs1 = char_seq(numpy_input)
        outputs2 = char_seq2(numpy_input)
        outputs3 = char_seq_diff_seed(numpy_input)

        self.assertEqual(outputs1.tolist(), outputs2.tolist())
        self.assertNotEqual(outputs1.tolist(), outputs3.tolist())
