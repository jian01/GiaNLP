"""
Trainable mapping embedding tests
"""

import unittest

from gianlp.models import MappingEmbedding, BaseModel


class TestTrainableWordEmbeddingSequence(unittest.TestCase):
    """
    Trainable mapping embedding tests
    """

    def test_shapes(self) -> None:
        """
        Test shapes
        """
        emb = MappingEmbedding("tests/resources/test_word2vec.txt")
        self.assertEqual(emb.inputs_shape.shape, (1,))
        self.assertEqual(emb.outputs_shape.shape, (3,))

    def test_known_vectors(self) -> None:
        """
        Test known vector assignment
        """
        emb = MappingEmbedding("tests/resources/test_word2vec.txt")
        emb.build([""])
        preproc = emb.preprocess_texts(["hola", "de", "que", "no"])
        preproc_embs = emb(preproc)
        for pred, real in zip(preproc_embs[0].tolist(), [0.34, -1.54, 0.22]):
            self.assertAlmostEqual(pred, real, delta=0.01)
        for pred, real in zip(preproc_embs[1].tolist(), [9.65, -1.23, 1.2]):
            self.assertAlmostEqual(pred, real, delta=0.01)
        for pred, real in zip(preproc_embs[2].tolist(), [6.5, 2.1, -4]):
            self.assertAlmostEqual(pred, real, delta=0.01)
        for pred, real in zip(preproc_embs[3].tolist(), [1.3, -0.02, 0.01]):
            self.assertAlmostEqual(pred, real, delta=0.01)

    def test_unknown_vectors(self) -> None:
        """
        Test unknown vector assignment
        """
        emb = MappingEmbedding("tests/resources/test_word2vec.txt")
        emb.build([""])
        preproc = emb.preprocess_texts(["coso1", "coso2"])
        preproc_embs = emb(preproc)
        self.assertEqual(preproc_embs[0].tolist(), preproc_embs[1].tolist())

    def test_serialization(self) -> None:
        """
        Test serialization
        """
        emb = MappingEmbedding("tests/resources/test_word2vec.txt")
        emb.build([""])
        data = emb.serialize()
        emb = BaseModel.deserialize(data)
        preproc = emb.preprocess_texts(["hola", "de", "que", "no", "coso1", "coso2"])
        preproc_embs = emb(preproc)
        for pred, real in zip(preproc_embs[0].tolist(), [0.34, -1.54, 0.22]):
            self.assertAlmostEqual(pred, real, delta=0.01)
        for pred, real in zip(preproc_embs[1].tolist(), [9.65, -1.23, 1.2]):
            self.assertAlmostEqual(pred, real, delta=0.01)
        for pred, real in zip(preproc_embs[2].tolist(), [6.5, 2.1, -4]):
            self.assertAlmostEqual(pred, real, delta=0.01)
        for pred, real in zip(preproc_embs[3].tolist(), [1.3, -0.02, 0.01]):
            self.assertAlmostEqual(pred, real, delta=0.01)
        self.assertEqual(preproc_embs[4].tolist(), preproc_embs[5].tolist())
