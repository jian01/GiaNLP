"""
RNN Digest tests
"""

import unittest

from gianlp.models import RNNDigest, CharEmbeddingSequence, PerChunkSequencer
from tests.utils import LOREM_IPSUM, newline_chunker


class TestRNNDigest(unittest.TestCase):
    """
    Simple test for RNN digest
    """

    def test_simple_lstm_digest(self) -> None:
        """
        Test with a simple lstm digest
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = RNNDigest(char_emb, 10, "lstm", stacked_layers=2, bidirectional=False, masking=False)
        model.build(LOREM_IPSUM.split("\n"))
        self.assertEqual(model.outputs_shape.shape, (10,))

    def test_time_distributed_gru_digest(self) -> None:
        """
        Test with a automatically time distributed GRU digest
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)
        per_chunk_sequencer = PerChunkSequencer(char_emb, newline_chunker, 2)

        model = RNNDigest(per_chunk_sequencer, 10, "gru", stacked_layers=2, bidirectional=True)
        model.build(LOREM_IPSUM.split("\n"))
        self.assertEqual(model.outputs_shape.shape, (2, 20))

    def test_simple_lstm_digest_multiple_inputs(self) -> None:
        """
        Test with a simple lstm digest with multiple inputs
        """
        char_emb1 = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)
        char_emb2 = CharEmbeddingSequence(embedding_dimension=20, sequence_maxlen=10)

        model = RNNDigest([char_emb1, char_emb2], 10, "lstm", stacked_layers=2, bidirectional=False, masking=False)
        model.build(LOREM_IPSUM.split("\n"))
        self.assertEqual(model.outputs_shape.shape, (10,))

    def test_sequence_mismatch_exception(self) -> None:
        """
        Test sequence length mismatch exception
        """
        char_emb1 = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)
        char_emb2 = CharEmbeddingSequence(embedding_dimension=20, sequence_maxlen=15)

        with self.assertRaises(ValueError):
            RNNDigest([char_emb1, char_emb2], 10, "lstm", stacked_layers=2, bidirectional=False, masking=False)

    def test_simple_lstm_return_sequences(self) -> None:
        """
        Test return_sequences parameter
        """
        char_emb1 = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)
        char_emb2 = CharEmbeddingSequence(embedding_dimension=20, sequence_maxlen=10)

        model = RNNDigest([char_emb1, char_emb2], 10, "lstm", stacked_layers=2, bidirectional=False, masking=False, return_sequences=True)
        model.build(LOREM_IPSUM.split("\n"))
        self.assertEqual(model.outputs_shape.shape, (10, 10))
