"""
Trainable word embedding sequence tests
"""

import unittest

import numpy as np
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from scipy.spatial.distance import cosine

from gianlp.models import TrainableWordEmbeddingSequence, KerasWrapper, BaseModel
from tests.utils import split_tokenizer


class TestTrainableWordEmbeddingSequence(unittest.TestCase):
    """
    Trainable word embedding sequence tests
    """

    def test_dimension_exception(self) -> None:
        """
        Test dimension mismatch
        """
        with self.assertRaises(ValueError):
            TrainableWordEmbeddingSequence(split_tokenizer, 4, "tests/resources/test_word2vec.txt", sequence_maxlen=4)

    def test_shapes(self) -> None:
        """
        Test shapes
        """
        emb = TrainableWordEmbeddingSequence(split_tokenizer, 3, "tests/resources/test_word2vec.txt", sequence_maxlen=4)
        self.assertEqual(emb.inputs_shape.shape, (4,))
        self.assertEqual(emb.outputs_shape.shape, (4, 3))

    def test_zero_vectors(self) -> None:
        """
        Test zero vector assignment
        """
        emb = TrainableWordEmbeddingSequence(split_tokenizer, 3, "tests/resources/test_word2vec.txt", sequence_maxlen=4)
        emb.build(
            [
                "hola como va",
                "hola que te importa",
                "no nada mas preguntaba",
                "me volves a preguntar y te rompo la cara",
                "no era mi intención de molestarte",
                "no te quedes mirandome andate",
            ]
        )
        preproc = emb.preprocess_texts(["coso"])
        self.assertEqual(preproc[0][1:].tolist(), [0, 0, 0])
        self.assertGreater(preproc[0][0], 0)
        preproc_embs = emb(preproc)
        self.assertEqual(preproc_embs[0][1].tolist(), [0] * 3)
        self.assertEqual(preproc_embs[0][2].tolist(), [0] * 3)
        self.assertEqual(preproc_embs[0][3].tolist(), [0] * 3)

    def test_known_vectors(self) -> None:
        """
        Test known vector assignment
        """
        emb = TrainableWordEmbeddingSequence(
            split_tokenizer, 3, "tests/resources/test_word2vec.txt", sequence_maxlen=5, min_freq_percentile=0
        )
        emb.build(
            [
                "hola como va",
                "hola que te importa",
                "no nada mas preguntaba",
                "me volves a preguntar y te rompo la cara",
                "no era mi intención de molestarte",
                "no te quedes mirandome andate aca",
            ]
        )
        preproc = emb.preprocess_texts(["coso hola de que no"])
        unk_preproc = preproc[0][0]
        self.assertGreater(unk_preproc, preproc[0][1])
        self.assertGreater(unk_preproc, preproc[0][2])
        self.assertGreater(unk_preproc, preproc[0][3])
        self.assertGreater(unk_preproc, preproc[0][4])
        preproc_embs = emb(preproc)
        for pred, real in zip(preproc_embs[0][1].tolist(), [0.34, -1.54, 0.22]):
            self.assertAlmostEqual(pred, real, delta=0.01)
        for pred, real in zip(preproc_embs[0][2].tolist(), [9.65, -1.23, 1.2]):
            self.assertAlmostEqual(pred, real, delta=0.01)
        for pred, real in zip(preproc_embs[0][3].tolist(), [6.5, 2.1, -4]):
            self.assertAlmostEqual(pred, real, delta=0.01)
        for pred, real in zip(preproc_embs[0][4].tolist(), [1.3, -0.02, 0.01]):
            self.assertAlmostEqual(pred, real, delta=0.01)

    def test_new_vectors(self) -> None:
        """
        Test new vector assignment
        """
        emb = TrainableWordEmbeddingSequence(
            split_tokenizer, 3, "tests/resources/test_word2vec.txt", sequence_maxlen=10, min_freq_percentile=0
        )
        texts = [
            "hola como va",
            "hola que te importa",
            "no nada mas preguntaba",
            "me volves a preguntar y te rompo la cara",
            "no era mi intención de molestarte",
            "no te quedes mirandome andate aca",
        ]
        emb.build(texts)
        preproc = emb.preprocess_texts(texts)
        preproc_embs = emb(preproc)
        for embs, text in zip(preproc_embs.tolist(), texts):
            self.assertEqual(len(embs), 10)
            for i in range(len(split_tokenizer(text))):
                self.assertEqual(len(embs[i]), 3)
                self.assertNotEqual(embs[i], [0, 0, 0])
            for i in range(len(split_tokenizer(text)), 10):
                self.assertEqual(len(embs[i]), 3)
                self.assertEqual(embs[i], [0, 0, 0])

    def test_non_trainable_pre_trained_embedding(self) -> None:
        """
        Test that the pre trained embeddings are non trainable
        """
        emb = TrainableWordEmbeddingSequence(
            split_tokenizer, 3, "tests/resources/test_word2vec.txt", sequence_maxlen=10, min_freq_percentile=0
        )
        texts = [
            "hola como va",
            "hola que te importa",
            "no nada mas preguntaba",
            "me volves a preguntar y te rompo la cara",
            "no era mi intención de molestarte",
            "no te quedes mirandome andate aca",
        ]
        emb.build(texts)
        preproc = emb.preprocess_texts(["coso hola de que no"])
        preproc_embs = emb(preproc).tolist()[0]

        model = Sequential(
            [GlobalAveragePooling1D(input_shape=emb.outputs_shape.shape), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(emb, model)
        model.build(texts)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(texts, np.asarray([0, 1] * 3), batch_size=2, epochs=10)

        preproc2 = emb.preprocess_texts(["coso hola de que no"])
        preproc_embs2 = emb(preproc2).tolist()[0]

        self.assertEqual(preproc.tolist(), preproc2.tolist())
        self.assertEqual(preproc_embs[1], preproc_embs2[1])
        self.assertEqual(preproc_embs[2], preproc_embs2[2])
        self.assertEqual(preproc_embs[3], preproc_embs2[3])
        self.assertEqual(preproc_embs[4], preproc_embs2[4])

    def test_trainable_pre_trained_embedding(self) -> None:
        """
        Test that the pre trained embeddings are trainable when setted as trainable
        """
        emb = TrainableWordEmbeddingSequence(
            split_tokenizer,
            3,
            "tests/resources/test_word2vec.txt",
            sequence_maxlen=10,
            min_freq_percentile=0,
            pretrained_trainable=True,
        )
        texts = [
            "hola como va",
            "hola que te importa",
            "no nada mas preguntaba",
            "me volves a preguntar y te rompo la cara",
            "no era mi intención de molestarte",
            "no te quedes mirandome andate aca",
        ]
        emb.build(texts)
        preproc = emb.preprocess_texts(["coso hola de que no"])
        preproc_embs = emb(preproc).tolist()[0]

        model = Sequential(
            [GlobalAveragePooling1D(input_shape=emb.outputs_shape.shape), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(emb, model)
        model.build(texts)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(texts, np.asarray([0, 1] * 3), batch_size=2, epochs=10)

        preproc2 = emb.preprocess_texts(["coso hola de que no"])
        preproc_embs2 = emb(preproc2).tolist()[0]

        self.assertEqual(preproc.tolist(), preproc2.tolist())
        self.assertNotEqual(preproc_embs[1], preproc_embs2[1])
        self.assertNotEqual(preproc_embs[2], preproc_embs2[2])
        self.assertNotEqual(preproc_embs[3], preproc_embs2[3])
        self.assertNotEqual(preproc_embs[4], preproc_embs2[4])

    def test_unknown_embedding_wont_train_if_dont_appear(self) -> None:
        """
        Test that the unknown embedding can't be trained if it does not appear in fit texts
        """
        emb = TrainableWordEmbeddingSequence(
            split_tokenizer, 3, "tests/resources/test_word2vec.txt", sequence_maxlen=10, min_freq_percentile=0
        )
        texts = [
            "hola como va",
            "hola que te importa",
            "no nada mas preguntaba",
            "me volves a preguntar y te rompo la cara",
            "no era mi intención de molestarte",
            "no te quedes mirandome andate aca",
        ]
        emb.build(texts)
        preproc = emb.preprocess_texts(["coso hola de que no"])
        preproc_embs = emb(preproc).tolist()[0]

        model = Sequential(
            [GlobalAveragePooling1D(input_shape=emb.outputs_shape.shape), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(emb, model)
        model.build(texts)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(texts, np.asarray([0, 1] * 3), batch_size=2, epochs=10)

        preproc2 = emb.preprocess_texts(["coso hola de que no"])
        preproc_embs2 = emb(preproc2).tolist()[0]

        self.assertEqual(preproc.tolist(), preproc2.tolist())
        self.assertEqual(preproc_embs[0], preproc_embs2[0])

    def test_trainable_unknown_embedding(self) -> None:
        """
        Test that the unknown embedding is always trainable
        """
        emb = TrainableWordEmbeddingSequence(
            split_tokenizer, 3, "tests/resources/test_word2vec.txt", sequence_maxlen=10, min_freq_percentile=0
        )
        texts = [
            "hola como va",
            "hola que te importa",
            "no nada mas preguntaba",
            "me volves a preguntar y te rompo la cara",
            "no era mi intención de molestarte",
            "no te quedes mirandome andate aca",
        ]
        emb.build(texts)
        preproc = emb.preprocess_texts(["coso hola de que no"])
        preproc_embs = emb(preproc).tolist()[0]

        model = Sequential(
            [GlobalAveragePooling1D(input_shape=emb.outputs_shape.shape), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(emb, model)
        model.build(texts)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(texts + ["coso hola", "coso no"], np.asarray([0, 1] * 4), batch_size=2, epochs=10)

        preproc2 = emb.preprocess_texts(["coso hola de que no"])
        preproc_embs2 = emb(preproc2).tolist()[0]

        self.assertEqual(preproc.tolist(), preproc2.tolist())
        self.assertNotEqual(preproc_embs[0], preproc_embs2[0])

        emb = TrainableWordEmbeddingSequence(
            split_tokenizer,
            3,
            "tests/resources/test_word2vec.txt",
            sequence_maxlen=10,
            min_freq_percentile=0,
            pretrained_trainable=True,
        )
        texts = [
            "hola como va",
            "hola que te importa",
            "no nada mas preguntaba",
            "me volves a preguntar y te rompo la cara",
            "no era mi intención de molestarte",
            "no te quedes mirandome andate aca",
        ]
        emb.build(texts)
        preproc = emb.preprocess_texts(["coso hola de que no"])
        preproc_embs = emb(preproc).tolist()[0]

        model = Sequential(
            [GlobalAveragePooling1D(input_shape=emb.outputs_shape.shape), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(emb, model)
        model.build(texts)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(texts + ["coso hola", "coso no"], np.asarray([0, 1] * 4), batch_size=2, epochs=10)

        preproc2 = emb.preprocess_texts(["coso hola de que no"])
        preproc_embs2 = emb(preproc2).tolist()[0]

        self.assertEqual(preproc.tolist(), preproc2.tolist())
        self.assertNotEqual(preproc_embs[0], preproc_embs2[0])

    def test_serialization(self) -> None:
        """
        Test serialization
        """
        emb = TrainableWordEmbeddingSequence(
            split_tokenizer, 3, "tests/resources/test_word2vec.txt", sequence_maxlen=10, min_freq_percentile=0
        )
        texts = [
            "hola como va",
            "hola que te importa",
            "no nada mas preguntaba",
            "me volves a preguntar y te rompo la cara",
            "no era mi intención de molestarte",
            "no te quedes mirandome andate aca",
        ]
        emb.build(texts)
        preproc = emb.preprocess_texts(["coso hola de que no"])
        preproc_embs = emb(preproc).tolist()[0]

        model = Sequential(
            [GlobalAveragePooling1D(input_shape=emb.outputs_shape.shape), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(emb, model)
        model.build(texts)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(texts, np.asarray([0, 1] * 3), batch_size=2, epochs=10)

        data = emb.serialize()
        emb = BaseModel.deserialize(data)

        preproc2 = emb.preprocess_texts(["coso hola de que no"])
        preproc_embs2 = emb(preproc2).tolist()[0]

        self.assertEqual(preproc.tolist(), preproc2.tolist())
        self.assertEqual(preproc_embs[1], preproc_embs2[1])
        self.assertEqual(preproc_embs[2], preproc_embs2[2])
        self.assertEqual(preproc_embs[3], preproc_embs2[3])
        self.assertEqual(preproc_embs[4], preproc_embs2[4])

    def test_trainable_new_embeddings(self) -> None:
        """
        Test that the new embeddings are trainable
        """
        emb = TrainableWordEmbeddingSequence(
            split_tokenizer, 3, "tests/resources/test_word2vec.txt", sequence_maxlen=10, min_freq_percentile=0
        )
        texts = [
            "hola como va",
            "hola que te importa",
            "no nada mas preguntaba",
            "me volves a preguntar y te rompo la cara",
            "no era mi intención de molestarte",
            "no te quedes mirandome andate aca",
        ]
        emb.build(texts)
        preproc = emb.preprocess_texts(["coso importa nada mas intención"])
        preproc_embs = emb(preproc).tolist()[0]

        model = Sequential(
            [GlobalAveragePooling1D(input_shape=emb.outputs_shape.shape), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(emb, model)
        model.build(texts)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(texts, np.asarray([0, 1] * 3), batch_size=2, epochs=10)

        preproc2 = emb.preprocess_texts(["coso importa nada mas intención"])
        preproc_embs2 = emb(preproc2).tolist()[0]

        self.assertEqual(preproc.tolist(), preproc2.tolist())
        self.assertNotEqual(preproc_embs[1], preproc_embs2[1])
        self.assertNotEqual(preproc_embs[2], preproc_embs2[2])
        self.assertNotEqual(preproc_embs[3], preproc_embs2[3])
        self.assertNotEqual(preproc_embs[4], preproc_embs2[4])

    def test_all_trainable_no_pretrained(self) -> None:
        """
        Test that all the embeddings are trainable if there are no pretrained ones
        """
        emb = TrainableWordEmbeddingSequence(split_tokenizer, 3, None, sequence_maxlen=10, min_freq_percentile=0)
        texts = [
            "hola como va",
            "hola que te importa",
            "no nada mas preguntaba",
            "me volves a preguntar y te rompo la cara",
            "no era mi intención de molestarte",
            "no te quedes mirandome andate aca",
        ]
        emb.build(texts)
        preproc = emb.preprocess_texts(["coso hola nada mas no"])
        preproc_embs = emb(preproc).tolist()[0]

        model = Sequential(
            [GlobalAveragePooling1D(input_shape=emb.outputs_shape.shape), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(emb, model)
        model.build(texts)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(texts, np.asarray([0, 1] * 3), batch_size=2, epochs=10)

        preproc2 = emb.preprocess_texts(["coso hola nada mas no"])
        preproc_embs2 = emb(preproc2).tolist()[0]

        self.assertEqual(preproc.tolist(), preproc2.tolist())
        self.assertNotEqual(preproc_embs[1], preproc_embs2[1])
        self.assertNotEqual(preproc_embs[2], preproc_embs2[2])
        self.assertNotEqual(preproc_embs[3], preproc_embs2[3])
        self.assertNotEqual(preproc_embs[4], preproc_embs2[4])

    def test_no_valid_vector_is_0(self):
        """
        Test that al valid vectors are != 0
        """
        emb = TrainableWordEmbeddingSequence(
            split_tokenizer, 3, "tests/resources/test_word2vec.txt", sequence_maxlen=10, min_freq_percentile=0
        )
        texts = [
            "hola como va",
            "hola que te importa",
            "no nada mas preguntaba",
            "me volves a preguntar y te rompo la cara",
            "no era mi intención de molestarte",
            "no te quedes mirandome andate aca",
        ]
        emb.build(texts)

        model = Sequential(
            [GlobalAveragePooling1D(input_shape=emb.outputs_shape.shape), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(emb, model)
        model.build(texts)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(texts, np.asarray([0, 1] * 3), batch_size=2, epochs=10)

        inference_texts = texts + ["coso hola nada mas no"]
        preproc = emb.preprocess_texts(inference_texts)
        preproc_embs = emb(preproc).tolist()

        for embs, text in zip(preproc_embs, inference_texts):
            for i in range(len(split_tokenizer(text))):
                self.assertNotEqual(embs[i], [0, 0, 0])
            for i in range(len(split_tokenizer(text)), 10):
                self.assertEqual(embs[i], [0, 0, 0])

    def test_max_vocabulary(self) -> None:
        """
        Test that the max vocabulary parameter works
        """
        emb = TrainableWordEmbeddingSequence(
            split_tokenizer,
            3,
            "tests/resources/test_word2vec.txt",
            sequence_maxlen=10,
            min_freq_percentile=0,
            max_vocabulary=6,
        )
        texts = [
            "hola como va",
            "hola que te importa",
            "no nada mas preguntaba",
            "me volves a preguntar y te rompo la cara",
            "no era mi intención de molestarte",
            "no te quedes mirandome andate aca",
        ]
        emb.build(texts)

        unk_vector = emb(emb.preprocess_texts(["coso"])).tolist()[0][0]

        corpus_preproc = emb(emb.preprocess_texts(texts)).tolist()

        not_unknown_tokens = set()

        for embs, text in zip(corpus_preproc, texts):
            for i in range(len(split_tokenizer(text))):
                if cosine(unk_vector, embs[i]) <= 0.01:
                    continue
                else:
                    not_unknown_tokens.update([split_tokenizer(text)[i]])

        self.assertEqual(len(not_unknown_tokens), 6)

    def test_random_seed(self) -> None:
        """
        Test random seed
        """
        texts = [
            "hola como va",
            "hola que te importa",
            "no nada mas preguntaba",
            "me volves a preguntar y te rompo la cara",
            "no era mi intención de molestarte",
            "no te quedes mirandome andate aca",
        ]
        word_emb = TrainableWordEmbeddingSequence(
            split_tokenizer,
            3,
            "tests/resources/test_word2vec.txt",
            sequence_maxlen=10,
            min_freq_percentile=0,
            random_state=12,
        )
        word_emb.build(texts)
        word_emb2 = TrainableWordEmbeddingSequence(
            split_tokenizer,
            3,
            "tests/resources/test_word2vec.txt",
            sequence_maxlen=10,
            min_freq_percentile=0,
            random_state=12,
        )
        word_emb2.build(texts)
        word_emb_diff_seed = TrainableWordEmbeddingSequence(
            split_tokenizer,
            3,
            "tests/resources/test_word2vec.txt",
            sequence_maxlen=10,
            min_freq_percentile=0,
            random_state=33,
        )
        word_emb_diff_seed.build(texts)

        outputs1 = word_emb(word_emb.preprocess_texts(["coso era rompo"]))
        outputs2 = word_emb2(word_emb2.preprocess_texts(["coso era rompo"]))
        outputs3 = word_emb_diff_seed(word_emb_diff_seed.preprocess_texts(["coso era rompo"]))

        self.assertEqual(outputs1.tolist(), outputs2.tolist())
        self.assertNotEqual(outputs1.tolist(), outputs3.tolist())

    def test_shapes_wout_pretrained(self) -> None:
        """
        Test shapes without pretrained embeddings
        """
        emb = TrainableWordEmbeddingSequence(split_tokenizer, 3, sequence_maxlen=4)
        self.assertEqual(emb.inputs_shape.shape, (4,))
        self.assertEqual(emb.outputs_shape.shape, (4, 3))

    def test_zero_vectors_wout_pretrained(self) -> None:
        """
        Test zero vector assignment without pretrained embeddings
        """
        emb = TrainableWordEmbeddingSequence(split_tokenizer, 3, sequence_maxlen=4)
        emb.build(
            [
                "hola como va",
                "hola que te importa",
                "no nada mas preguntaba",
                "me volves a preguntar y te rompo la cara",
                "no era mi intención de molestarte",
                "no te quedes mirandome andate",
            ]
        )
        preproc = emb.preprocess_texts(["coso"])
        self.assertEqual(preproc[0][1:].tolist(), [0, 0, 0])
        self.assertGreater(preproc[0][0], 0)
        preproc_embs = emb(preproc)
        self.assertEqual(preproc_embs[0][1].tolist(), [0] * 3)
        self.assertEqual(preproc_embs[0][2].tolist(), [0] * 3)
        self.assertEqual(preproc_embs[0][3].tolist(), [0] * 3)
