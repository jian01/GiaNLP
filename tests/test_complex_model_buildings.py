"""
Test complex model buildings
"""

import unittest

import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, GRU, Dense, Masking, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.models import Sequential, Model

from gianlp.models import KerasWrapper, BaseModel, CharEmbeddingSequence, PerChunkSequencer
from tests.utils import dot_chunker, LOREM_IPSUM, set_seed, read_sms_spam_dset


class TestComplexModelBuildings(unittest.TestCase):
    """
    Tests for complex model architectures
    """

    def test_fit_over_multiple_inputs(self) -> None:
        """
        Test fiting with multiple inputs and models chained
        """
        set_seed(42)

        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=128)
        char_digest = Sequential(
            [
                Input(char_emb.outputs_shape.shape),
                Masking(0.0),
                GRU(20, activation="tanh"),
                Dense(20, activation="tanh"),
            ]
        )
        char_digest = KerasWrapper(char_emb, char_digest)

        char_emb2 = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=128)
        per_chunk_sequencer = PerChunkSequencer(char_emb2, dot_chunker, chunking_maxlen=10)
        word_digestor = Sequential(
            [
                Input(char_emb.outputs_shape.shape),
                Masking(0.0),
                GRU(10, activation="tanh"),
                Dense(10, activation="tanh"),
            ]
        )
        word_digestor = KerasWrapper(per_chunk_sequencer, word_digestor)
        line_digest = Sequential([Input((10, 10)), GlobalMaxPooling1D()])
        line_digest = KerasWrapper(word_digestor, line_digest)

        model = Sequential([Input((30,)), Dense(1, activation="sigmoid")])
        model = KerasWrapper([char_digest, line_digest], model)

        texts, labels = read_sms_spam_dset()
        model.build(texts)
        print(model)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        hst = model.fit(texts, np.asarray(labels), batch_size=256, epochs=6, validation_split=0.1)
        self.assertAlmostEqual(hst.history["accuracy"][-1], 1.0, delta=0.1)
        self.assertAlmostEqual(hst.history["val_accuracy"][-1], 1.0, delta=0.1)
        preds1 = model.predict(["prueba 1", "prueba. prueba 2"])
        serialized = model.serialize()
        try:
            model2: KerasWrapper = BaseModel.deserialize(serialized)
        except NotImplementedError:
            return  # tensorflow lower versions known issue
        preds2 = model2.predict(["prueba 1", "prueba. prueba 2"])
        self.assertEqual(preds1.tolist(), preds2.tolist())

    def test_fit_over_multiple_inputs_pandas_series(self) -> None:
        """
        Test fiting with multiple inputs and models chained, using a pandas series as texts inputs
        """
        set_seed(42)

        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=128)
        char_digest = Sequential(
            [
                Input(char_emb.outputs_shape.shape),
                Masking(0.0),
                GRU(20, activation="tanh"),
                Dense(20, activation="tanh"),
            ]
        )
        char_digest = KerasWrapper(char_emb, char_digest)

        char_emb2 = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=128, min_freq_percentile=0)
        per_chunk_sequencer = PerChunkSequencer(char_emb2, dot_chunker, chunking_maxlen=10)
        word_digestor = Sequential(
            [
                Input(char_emb.outputs_shape.shape),
                Masking(0.0),
                GRU(10, activation="tanh"),
                Dense(10, activation="tanh"),
            ]
        )
        word_digestor = KerasWrapper(per_chunk_sequencer, word_digestor)
        line_digest = Sequential([Input((10, 10)), GlobalMaxPooling1D()])
        line_digest = KerasWrapper(word_digestor, line_digest)

        model = Sequential([Input((30,)), Dense(1, activation="sigmoid")])
        model = KerasWrapper([char_digest, line_digest], model)

        texts, labels = read_sms_spam_dset()
        model.build(texts)
        print(model)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        texts_series = pd.Series(texts)
        np.random.seed(100)
        texts_series.index = texts_series.index.values[np.random.permutation(len(texts_series))]
        hst = model.fit(texts_series, np.asarray(labels), batch_size=256, epochs=6, validation_split=0.1)
        self.assertAlmostEqual(hst.history["accuracy"][-1], 1.0, delta=0.1)
        self.assertAlmostEqual(hst.history["val_accuracy"][-1], 1.0, delta=0.1)
        preds1 = model.predict(["prueba 1", "prueba. prueba 2"])
        serialized = model.serialize()
        try:
            model2: KerasWrapper = BaseModel.deserialize(serialized)
        except NotImplementedError:
            return  # tensorflow lower versions known issue
        preds2 = model2.predict(["prueba 1", "prueba. prueba 2"])
        self.assertEqual(preds1.tolist(), preds2.tolist())

    def test_simple_layer_reusing(self) -> None:
        """
        Test layer reuse with serialization and deserialization
        """
        set_seed(42)

        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=128)
        char_digest = Sequential(
            [
                Input(char_emb.outputs_shape.shape),
                Masking(0.0),
                GRU(20, activation="tanh"),
                Dense(20, activation="tanh"),
            ]
        )
        char_digest = KerasWrapper(char_emb, char_digest)

        per_chunk_sequencer = PerChunkSequencer(char_emb, dot_chunker, chunking_maxlen=10)
        word_digestor = Sequential(
            [
                Input(char_emb.outputs_shape.shape),
                Masking(0.0),
                GRU(10, activation="tanh"),
                Dense(10, activation="tanh"),
            ]
        )
        word_digestor = KerasWrapper(per_chunk_sequencer, word_digestor)
        line_digest = Sequential([Input((10, 10)), GlobalMaxPooling1D()])
        line_digest = KerasWrapper(word_digestor, line_digest)

        inp1 = Input((20,))
        inp2 = Input((10,))
        concat = Concatenate()([inp1, inp2])
        out = Dense(1, activation="sigmoid")(concat)
        model = Model(inputs=[inp1, inp2], outputs=out)
        model = KerasWrapper([char_digest, line_digest], model)

        texts, labels = read_sms_spam_dset()
        model.build(texts)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        preds1 = model.predict(["prueba 1", "prueba. prueba 2"])
        serialized = model.serialize()
        try:
            model2: KerasWrapper = BaseModel.deserialize(serialized)
        except NotImplementedError:
            return  # tensorflow lower versions known issue
        preds2 = model2.predict(["prueba 1", "prueba. prueba 2"])
        self.assertEqual(preds1.tolist(), preds2.tolist())

    def test_siamese_layer_reusing(self) -> None:
        """
        Test layer reuse with serialization and deserialization for a siamese architecture
        """
        set_seed(42)

        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(10, activation="tanh")]
        )
        encoder = KerasWrapper(char_emb, model)

        siamese = Sequential([Input((20,)), Dense(1, activation="sigmoid")])
        siamese = KerasWrapper([encoder, encoder], siamese)
        siamese.build(LOREM_IPSUM.split(" "))

        preds1 = siamese.predict(["prueba 1", "prueba. prueba 2"])
        serialized = siamese.serialize()
        siamese2: KerasWrapper = BaseModel.deserialize(serialized)
        preds2 = siamese2.predict(["prueba 1", "prueba. prueba 2"])
        self.assertEqual(preds1.tolist(), preds2.tolist())
