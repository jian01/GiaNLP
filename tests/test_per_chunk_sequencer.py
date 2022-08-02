"""
Per chunk sequencer tests
"""

import unittest

import numpy as np
from tensorflow.keras.layers import Input

from gianlp.models import BaseModel
from gianlp.models import CharPerWordEmbeddingSequence
from gianlp.models import PerChunkSequencer
from tests.utils import split_tokenizer, LOREM_IPSUM, newline_chunker


class TestPerChunkSequencer(unittest.TestCase):
    """
    Pre trained per chunk sequencer
    """

    @staticmethod
    def create_test_obj(random_seed: int = 42) -> PerChunkSequencer:
        """
        Creates a test embedding
        """
        char_emb = CharPerWordEmbeddingSequence(
            tokenizer=split_tokenizer, word_maxlen=4, char_maxlen=5, random_state=random_seed
        )
        per_chunk_sequencer = PerChunkSequencer(char_emb, newline_chunker, 2)
        per_chunk_sequencer.build(LOREM_IPSUM.split("\n"))
        return per_chunk_sequencer

    def test_preprocess(self) -> None:
        """
        Test preprocess method
        """
        per_chunk_sequencer = self.create_test_obj()
        preprocessed1 = per_chunk_sequencer.preprocess_texts(["prueba das\notra cosa"])
        preprocessed2 = per_chunk_sequencer.preprocess_texts(["prueba das\notra cosa"])
        self.assertEqual(preprocessed1.tolist(), preprocessed2.tolist())

    def test_maxlen(self) -> None:
        """
        Test maxlen of a sequence
        """
        per_chunk_sequencer = self.create_test_obj()
        preprocessed = per_chunk_sequencer.preprocess_texts(["prueba hola prueba"])
        self.assertTrue((preprocessed > 0).sum() == 14)
        self.assertTrue(not preprocessed[3:].any())
        preprocessed = per_chunk_sequencer.preprocess_texts(["prueba prueba hola prueba prueba"])
        self.assertTrue((preprocessed > 0).sum() == 19)
        self.assertEqual(preprocessed.shape, (1, 2, 4, 5))
        preprocessed = per_chunk_sequencer.preprocess_texts(["prueba prueba hola prueba prueba\nprueba"])
        self.assertTrue((preprocessed > 0).sum() == 24)
        self.assertEqual(preprocessed.shape, (1, 2, 4, 5))
        preprocessed = per_chunk_sequencer.preprocess_texts(["prueba prueba hola prueba prueba\nprueba\nprueba"])
        self.assertTrue((preprocessed > 0).sum() == 24)
        self.assertEqual(preprocessed.shape, (1, 2, 4, 5))

    def test_serialize(self) -> None:
        """
        Test object serialization
        """
        per_chunk_sequencer: PerChunkSequencer = self.create_test_obj()
        preprocessed1 = per_chunk_sequencer.preprocess_texts(["hola"])
        data = per_chunk_sequencer.serialize()
        per_chunk_sequencer = BaseModel.deserialize(data)
        preprocessed2 = per_chunk_sequencer.preprocess_texts(["hola"])
        self.assertEqual(preprocessed1.tolist(), preprocessed2.tolist())

    def test_functional_api_simple_layer(self) -> None:
        """
        Test object functional api
        """
        per_chunk_sequencer = self.create_test_obj()
        inputs = Input(shape=(2, 4, 5))
        outputs = per_chunk_sequencer(inputs)
        self.assertEqual(outputs.shape.as_list(), [None, 2, 4, 5, 256])

    def test_functional_api_multiple_inputs(self) -> None:
        """
        Test object functional api with multiple inputs
        """
        per_chunk_sequencer = self.create_test_obj()
        inputs1 = Input(shape=(2, 4, 5))
        inputs2 = Input(shape=(2, 4, 5))
        outputs = per_chunk_sequencer([inputs1, inputs2])
        self.assertEqual(outputs.shape.as_list(), [None, 2, 4, 5, 512])

    def test_functional_api_multiple_sequence(self) -> None:
        """
        Test object functional api with sequence with more dims
        """
        per_chunk_sequencer = self.create_test_obj()
        inputs = Input(shape=(100, 2, 4, 5))
        outputs = per_chunk_sequencer(inputs)
        self.assertEqual(outputs.shape.as_list(), [None, 100, 2, 4, 5, 256])
        inputs = Input(shape=(100, 12, 2, 4, 5))
        outputs = per_chunk_sequencer(inputs)
        self.assertEqual(outputs.shape.as_list(), [None, 100, 12, 2, 4, 5, 256])

    def test_functional_api_multi_sequence_inputs(self) -> None:
        """
        Test object functional api complex scenario
        """
        per_chunk_sequencer = self.create_test_obj()
        inputs1 = Input(shape=(100, 2, 4, 5))
        inputs2 = Input(shape=(100, 2, 4, 5))
        outputs = per_chunk_sequencer([inputs1, inputs2])
        self.assertEqual(outputs.shape.as_list(), [None, 100, 2, 4, 5, 512])

    def test_functional_api_numpy_sequence(self) -> None:
        """
        Test object functional api
        """
        per_chunk_sequencer = self.create_test_obj()
        numpy_input = np.random.randint(0, 10, size=(1, 2, 4, 5))
        outputs = per_chunk_sequencer(numpy_input)
        self.assertEqual(outputs.shape, (1, 2, 4, 5, 256))
