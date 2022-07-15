"""
Tests for multiple model outputs
"""

import unittest
from typing import List, Tuple

import numpy as np
from tensorflow.keras.layers import Input, GRU, Dense, Masking, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential, Model
from tensorflow.random import set_seed

from gianlp.models import KerasWrapper, CharEmbeddingSequence, PerChunkSequencer
from tests.utils import dot_chunker


class TestMultipleOutputs(unittest.TestCase):
    """
    Tests for multiple model outputs
    """

    @staticmethod
    def read_sms_spam_dset() -> Tuple[List[str], List[int]]:
        """
        Reads and returns sms spam dataset
        :return: a tuple containing a list with the text and a list with the labels
        """
        texts = []
        labels = []
        with open("tests/resources/SMSSpamCollection.txt", "r") as file:
            for line in file:
                if line:
                    line = line.split("\t")
                    texts.append(line[1])
                    labels.append((1 if line[0] == "spam" else 0))
        return texts, labels

    def test_fit_two_outputs(self) -> None:
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

        inp = Input((30,))
        out1 = Dense(1, activation="sigmoid")(inp)
        out2 = Dense(1, activation="sigmoid")(inp)
        model = Model(inputs=inp, outputs=[out1, out2])
        model = KerasWrapper([char_digest, line_digest], model)

        texts, labels = self.read_sms_spam_dset()
        model.build(texts)
        print(model)
        model.compile(optimizer="adam", loss="mean_absolute_error")
        hst = model.fit(
            texts, [np.asarray(labels), 1 - np.asarray(labels)], batch_size=256, epochs=2, validation_split=0.1
        )
        self.assertLess(hst.history["loss"][-1], 0.5)
        self.assertLess(hst.history["val_loss"][-1], 0.5)
        mean_error = sum(abs(model.predict(texts)[0].flatten() - np.asarray(labels))) / len(texts)
        self.assertLess(mean_error, 0.5)
        mean_error = sum(abs(model.predict(texts)[1].flatten() - (1 - np.asarray(labels)))) / len(texts)
        self.assertLess(mean_error, 0.5)
