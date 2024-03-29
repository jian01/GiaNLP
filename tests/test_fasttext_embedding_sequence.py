"""
Fasttext embedding sequence tests
"""

import unittest

import numpy as np
from scipy.spatial.distance import cosine
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from gensim.models.fasttext import load_facebook_model

from gianlp.models import FasttextEmbeddingSequence, KerasWrapper, BaseModel
from tests.utils import split_tokenizer


class TestTrainableWordEmbeddingSequence(unittest.TestCase):
    """
    Fasttext embedding sequence tests
    """

    def test_shapes(self) -> None:
        """
        Test shapes
        """
        emb = FasttextEmbeddingSequence(split_tokenizer, "tests/resources/fasttext.bin", sequence_maxlen=4)
        self.assertEqual(emb.inputs_shape.shape, (4,))
        self.assertEqual(emb.outputs_shape.shape, (4, 2))

    def test_fasttext_as_parameter(self) -> None:
        """
        Test shapes
        """
        emb = FasttextEmbeddingSequence(
            split_tokenizer, load_facebook_model("tests/resources/fasttext.bin"), sequence_maxlen=4
        )
        self.assertEqual(emb.inputs_shape.shape, (4,))
        self.assertEqual(emb.outputs_shape.shape, (4, 2))

    def test_zero_vectors(self) -> None:
        """
        Test zero vector assignment
        """
        emb = FasttextEmbeddingSequence(split_tokenizer, "tests/resources/fasttext.bin", sequence_maxlen=4)
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
        self.assertEqual(preproc_embs[0][1].tolist(), [0] * 2)
        self.assertEqual(preproc_embs[0][2].tolist(), [0] * 2)
        self.assertEqual(preproc_embs[0][3].tolist(), [0] * 2)

    def test_unknown_embedding_wont_train_if_dont_appear(self) -> None:
        """
        Test that the unknown embedding can't be trained if it does not appear in fit texts
        """
        emb = FasttextEmbeddingSequence(split_tokenizer, "tests/resources/fasttext.bin", sequence_maxlen=10)
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
        emb = FasttextEmbeddingSequence(split_tokenizer, "tests/resources/fasttext.bin", sequence_maxlen=10)
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
        emb = FasttextEmbeddingSequence(split_tokenizer, "tests/resources/fasttext.bin", sequence_maxlen=10)
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

    def test_no_valid_vector_is_0(self):
        """
        Test that al valid vectors are != 0
        """
        emb = FasttextEmbeddingSequence(split_tokenizer, "tests/resources/fasttext.bin", sequence_maxlen=10)
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
                self.assertNotEqual(embs[i], [0, 0])
            for i in range(len(split_tokenizer(text)), 10):
                self.assertEqual(embs[i], [0, 0])

    def test_max_vocabulary(self) -> None:
        """
        Test that the max vocabulary parameter works
        """
        emb = FasttextEmbeddingSequence(
            split_tokenizer, "tests/resources/fasttext.bin", sequence_maxlen=10, max_vocabulary=6
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
