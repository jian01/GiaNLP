"""
Tests for multiprocessing
"""

import unittest
from typing import List, Tuple
import string
import random

import numpy as np
from tensorflow.keras.layers import Input, GRU, Dense, Masking, GlobalMaxPooling1D, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.random import set_seed

from gianlp.models import KerasWrapper, BaseModel, CharEmbeddingSequence, PerChunkSequencer
from gianlp.utils import Sequence
from tests.utils import dot_chunker
from tests.utils import LOREM_IPSUM, accuracy, generator_from_list, SequenceFromList


class TestSpamSequence(Sequence):
    """
    Spam text sequence
    """

    def __init__(self, texts, labels, batch_size):
        self.texts = texts
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.texts) // 256

    def __getitem__(self, index: int):
        texts = self.texts[index * self.batch_size : (index + 1) * self.batch_size]
        labels = self.labels[index * self.batch_size : (index + 1) * self.batch_size]

        return texts, np.asarray(labels)


class TestMultiprocessing(unittest.TestCase):
    """
    Tests for multiprocessing
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

    def test_fit_with_sequence_valid_with_raw(self) -> None:
        """
        Test fiting using sequence validating using the raw data
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

        texts, labels = self.read_sms_spam_dset()
        val_texts, val_labels = texts[-500:], labels[-500:]
        texts, labels = texts[:-500], labels[:-500]
        model.build(texts)
        sequence_x = TestSpamSequence(texts, labels, batch_size=256)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        hst = model.fit(
            sequence_x,
            epochs=6,
            steps_per_epoch=len(texts) // 256,
            validation_data=(val_texts, np.asarray(val_labels)),
            max_queue_size=10,
            workers=2,
            use_multiprocessing=True,
        )
        self.assertAlmostEqual(hst.history["accuracy"][-1], 1.0, delta=0.1)
        self.assertAlmostEqual(hst.history["val_accuracy"][-1], 1.0, delta=0.1)
        preds1 = model.predict(["prueba 1", "prueba. prueba 2"])
        serialized = model.serialize()
        model2: KerasWrapper = BaseModel.deserialize(serialized)
        preds2 = model2.predict(["prueba 1", "prueba. prueba 2"])
        self.assertEqual(preds1.tolist(), preds2.tolist())

    def test_multiprocessing_w_data_warning(self) -> None:
        """
        Test multiprocessing with data gives warning but works
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

        texts, labels = self.read_sms_spam_dset()
        val_texts, val_labels = texts[-500:], labels[-500:]
        texts, labels = texts[:-500], labels[:-500]
        model.build(texts)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(
            texts,
            np.asarray(labels),
            epochs=1,
            steps_per_epoch=len(texts) // 256,
            validation_data=(val_texts, np.asarray(val_labels)),
            max_queue_size=10,
            workers=2,
            use_multiprocessing=True,
        )

    @staticmethod
    def starts_with_vocal_generator():
        """
        Generator for training that generates texts with label 1 if starts with vocal
        """
        letter_with_no_vocals = string.ascii_uppercase
        vocals = ["A", "E", "I", "O", "U"]
        for v in vocals:
            letter_with_no_vocals = letter_with_no_vocals.replace(v, "")
        while True:
            texts = []
            labels = []
            for i in range(64):
                texts += [
                    "".join(
                        np.random.choice(list(letter_with_no_vocals) + list(string.digits))
                        for _ in range(np.random.randint(2, 8))
                    )
                ]
            for i in range(64):
                if random.randint(0, 1) == 1:
                    texts[i] = texts[i][1:] + np.random.choice(vocals)
                    labels.append(1)
                else:

                    labels.append(0)
            yield texts, np.asarray(labels)

    def test_classifier_with_generators(self) -> None:
        """
        Training with generators and multiprocessing
        """
        set_seed(42)
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        gru_digest = Sequential([Input(char_emb.outputs_shape.shape), Masking(0.0), GRU(10, activation="tanh")])
        gru_digest = KerasWrapper(char_emb, gru_digest)

        cnn_digest = Sequential(
            [Input(char_emb.outputs_shape.shape), Conv1D(10, 1, activation="tanh"), GlobalMaxPooling1D()]
        )
        cnn_digest = KerasWrapper(char_emb, cnn_digest)

        model = Sequential([Input((20,)), Dense(1, activation="sigmoid")])
        model = KerasWrapper([gru_digest, cnn_digest], model)

        model.build(LOREM_IPSUM.split("\n"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        hst = model.fit(
            self.starts_with_vocal_generator(),
            epochs=30,
            steps_per_epoch=100,
            validation_data=self.starts_with_vocal_generator(),
            validation_steps=10,
            use_multiprocessing=True,
            workers=2,
            max_queue_size=10,
        )
        self.assertAlmostEqual(hst.history["accuracy"][-1], 1.0, delta=0.15)
        self.assertAlmostEqual(hst.history["val_accuracy"][-1], 1.0, delta=0.15)
        preds = model.predict(["A", "E", "I", "O", "U", "JS4DS", "S4DS", "4DS", "DS", "S"])
        preds = [1 if p > 0.5 else 0 for p in preds.flatten()]
        self.assertAlmostEqual(accuracy([1] * 5 + [0] * 5, preds), 1.0, delta=0.15)

    def test_predict_with_parallel_generator_allowed(self) -> None:
        """
        Prediction with paralellized generator is allowed although not recommended
        """
        set_seed(42)
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        gru_digest = Sequential([Input(char_emb.outputs_shape.shape), Masking(0.0), GRU(10, activation="tanh")])
        gru_digest = KerasWrapper(char_emb, gru_digest)

        cnn_digest = Sequential(
            [Input(char_emb.outputs_shape.shape), Conv1D(10, 1, activation="tanh"), GlobalMaxPooling1D()]
        )
        cnn_digest = KerasWrapper(char_emb, cnn_digest)

        model = Sequential([Input((20,)), Dense(1, activation="sigmoid")])
        model = KerasWrapper([gru_digest, cnn_digest], model)

        model.build(LOREM_IPSUM.split("\n"))
        model.predict(
            generator_from_list([["A"], ["E"], ["I"], ["O"], ["U"], ["JS4DS"], ["S4DS"], ["4DS"], ["DS"], ["S"]]),
            use_multiprocessing=True,
            workers=2,
            steps=10,
        )

    def test_predict_with_parallel_sequence_object(self) -> None:
        """
        Predict with parallel sequence object
        """
        set_seed(42)
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        gru_digest = Sequential([Input(char_emb.outputs_shape.shape), Masking(0.0), GRU(10, activation="tanh")])
        gru_digest = KerasWrapper(char_emb, gru_digest)

        cnn_digest = Sequential(
            [Input(char_emb.outputs_shape.shape), Conv1D(10, 1, activation="tanh"), GlobalMaxPooling1D()]
        )
        cnn_digest = KerasWrapper(char_emb, cnn_digest)

        model = Sequential([Input((20,)), Dense(1, activation="sigmoid")])
        model = KerasWrapper([gru_digest, cnn_digest], model)

        model.build(LOREM_IPSUM.split("\n"))
        preds = model.predict(["A", "E", "I", "O", "U", "JS4DS", "S4DS", "4DS", "DS", "S"]).round(3).astype("float32")
        preds2 = (
            model.predict(
                SequenceFromList([["A"], ["E"], ["I"], ["O"], ["U"], ["JS4DS"], ["S4DS"], ["4DS"], ["DS"], ["S"]])
            )
            .round(3)
            .astype("float32")
        )
        preds3 = (
            model.predict(
                SequenceFromList([["A"], ["E"], ["I"], ["O"], ["U"], ["JS4DS"], ["S4DS"], ["4DS"], ["DS"], ["S"]]),
                use_multiprocessing=True,
                workers=2,
            )
            .round(3)
            .astype("float32")
        )
        self.assertEqual(preds.tolist(), preds2.tolist())
        self.assertEqual(preds.tolist(), preds3.tolist())

    def test_predict_with_parallel_sequence__irregular_batches(self) -> None:
        """
        Predict with parallel sequence object using irregular batches
        """
        set_seed(42)
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        gru_digest = Sequential([Input(char_emb.outputs_shape.shape), Masking(0.0), GRU(10, activation="tanh")])
        gru_digest = KerasWrapper(char_emb, gru_digest)

        cnn_digest = Sequential(
            [Input(char_emb.outputs_shape.shape), Conv1D(10, 1, activation="tanh"), GlobalMaxPooling1D()]
        )
        cnn_digest = KerasWrapper(char_emb, cnn_digest)

        model = Sequential([Input((20,)), Dense(1, activation="sigmoid")])
        model = KerasWrapper([gru_digest, cnn_digest], model)

        model.build(LOREM_IPSUM.split("\n"))
        preds = model.predict(["A", "E", "I", "O", "U", "JS4DS", "S4DS", "4DS", "DS", "S"]).round(3).astype("float32")
        preds2 = (
            model.predict(SequenceFromList([["A"], ["E"], ["I"], ["O"], ["U", "JS4DS"], ["S4DS", "4DS", "DS"], ["S"]]))
            .round(3)
            .astype("float32")
        )
        preds3 = (
            model.predict(
                SequenceFromList([["A"], ["E"], ["I"], ["O"], ["U", "JS4DS"], ["S4DS", "4DS", "DS"], ["S"]]),
                use_multiprocessing=True,
                workers=2,
            )
            .round(3)
            .astype("float32")
        )
        self.assertEqual(preds.tolist(), preds2.tolist())
        self.assertEqual(preds.tolist(), preds3.tolist())
