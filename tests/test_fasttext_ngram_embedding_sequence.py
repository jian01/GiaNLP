"""
Fasttext embedding sequence utils
"""

import unittest

import numpy as np
from gensim.models.fasttext import load_facebook_model
from scipy.spatial.distance import cosine
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Input, Masking, GRU
from tensorflow.keras.models import Sequential

from gianlp.models import FasttextNgramEmbeddingSequence, KerasWrapper, BaseModel
from tests.utils import split_tokenizer, read_sms_spam_dset, set_seed


class TestFasttextNgramEmbedding(unittest.TestCase):
    """
    Fasttext embedding sequence utils
    """

    def test_shapes(self) -> None:
        """
        Test shapes
        """
        emb = FasttextNgramEmbeddingSequence(split_tokenizer, "tests/resources/fasttext.bin", sequence_maxlen=4)
        self.assertEqual(emb.inputs_shape.shape, (4, None))
        self.assertEqual(emb.outputs_shape.shape, (4, 50))

    def test_fasttext_as_parameter(self) -> None:
        """
        Test shapes
        """
        emb = FasttextNgramEmbeddingSequence(
            split_tokenizer, load_facebook_model("tests/resources/fasttext.bin"), sequence_maxlen=4
        )
        self.assertEqual(emb.inputs_shape.shape, (4, None))
        self.assertEqual(emb.outputs_shape.shape, (4, 50))

    def test_zero_vectors(self) -> None:
        """
        Test zero vector assignment
        """
        emb = FasttextNgramEmbeddingSequence(split_tokenizer, "tests/resources/fasttext.bin", sequence_maxlen=2)
        emb.build([""])
        preproc = emb.preprocess_texts([" ", ""]).to_list()
        self.assertEqual(preproc, [[[0], [0]], [[0], [0]]])
        preproc = emb.preprocess_texts(["asd"]).to_list()
        self.assertEqual(preproc[0][1], [0])

        self.assertEqual(emb(emb.preprocess_texts(["asd"])).tolist()[0][1], [0] * 50)

    def test_dynamic_vectors(self) -> None:
        """
        Test dynamic vector assignment
        """
        fasttext = load_facebook_model("tests/resources/fasttext.bin")
        emb = FasttextNgramEmbeddingSequence(split_tokenizer, fasttext, sequence_maxlen=4)
        emb.build([""])
        word_vec1 = emb(emb.preprocess_texts(['woman'])).tolist()[0][0]
        word_vec2 = fasttext.wv['woman'].tolist()
        word_vec3 = emb(emb.preprocess_texts(['man'])).tolist()[0][0]
        word_vec4 = fasttext.wv['man'].tolist()
        self.assertLessEqual(cosine(word_vec1, word_vec2), 0.005)
        self.assertGreaterEqual(cosine(word_vec3, word_vec1), 0.05)
        self.assertAlmostEqual(cosine(word_vec3, word_vec1), cosine(word_vec2, word_vec4), delta=0.005)
        self.assertLessEqual(cosine(word_vec3, word_vec4), 0.005)

    def test_normalize(self) -> None:
        """
        Test fasttext normalize on inference
        """
        fasttext = load_facebook_model("tests/resources/fasttext.bin")
        emb = FasttextNgramEmbeddingSequence(split_tokenizer, fasttext, sequence_maxlen=4)
        emb.build([""])
        emb_normalized = FasttextNgramEmbeddingSequence(split_tokenizer, fasttext,
                                                        sequence_maxlen=4, normalized=True)
        emb_normalized.build([""])
        word_vec1 = emb(emb.preprocess_texts(['woman'])).tolist()[0][0]
        word_vec2 = emb_normalized(emb_normalized.preprocess_texts(['woman'])).tolist()[0][0]
        word_vec3 = emb(emb.preprocess_texts(['man'])).tolist()[0][0]
        word_vec4 = emb_normalized(emb_normalized.preprocess_texts(['man'])).tolist()[0][0]

        self.assertLessEqual(cosine(word_vec1, word_vec2), 0.00005)
        self.assertLessEqual(cosine(word_vec3, word_vec4), 0.00005)
        self.assertGreaterEqual(cosine(word_vec3, word_vec1), 0.05)

        self.assertNotAlmostEqual(np.linalg.norm(word_vec1), 1, delta=0.05)
        self.assertAlmostEqual(np.linalg.norm(word_vec2), 1, delta=0.01)
        self.assertNotAlmostEqual(np.linalg.norm(word_vec3), 1, delta=0.05)
        self.assertAlmostEqual(np.linalg.norm(word_vec4), 1, delta=0.01)

    def test_irregular_sequences(self) -> None:
        """
        Test fasttext over irregular sequences
        """
        fasttext = load_facebook_model("tests/resources/fasttext.bin")
        emb = FasttextNgramEmbeddingSequence(split_tokenizer, fasttext, sequence_maxlen=4)
        emb.build([""])
        emb_normalized = FasttextNgramEmbeddingSequence(split_tokenizer, fasttext,
                                                        sequence_maxlen=4, normalized=True)
        emb_normalized.build([""])
        woman = emb(emb.preprocess_texts(['woman'])).tolist()[0][0]
        man = emb(emb.preprocess_texts(['man'])).tolist()[0][0]
        robot = emb(emb.preprocess_texts(['robot'])).tolist()[0][0]
        german = emb(emb.preprocess_texts(['german'])).tolist()[0][0]

        texts = emb(emb.preprocess_texts(['german robot',
                                          'man',
                                          'robot man',
                                          'german robot woman'])).tolist()

        self.assertLessEqual(cosine(texts[0][0], german), 0.00005)
        self.assertLessEqual(cosine(texts[0][1], robot), 0.00005)
        self.assertLessEqual(cosine(texts[1][0], man), 0.00005)
        self.assertLessEqual(cosine(texts[2][0], robot), 0.00005)
        self.assertLessEqual(cosine(texts[2][1], man), 0.00005)
        self.assertLessEqual(cosine(texts[3][0], german), 0.00005)
        self.assertLessEqual(cosine(texts[3][1], robot), 0.00005)
        self.assertLessEqual(cosine(texts[3][2], woman), 0.00005)

        self.assertEqual(len(texts[0]), 4)
        self.assertEqual(len(texts[1]), 4)
        self.assertEqual(len(texts[2]), 4)
        self.assertEqual(len(texts[3]), 4)

    def test_serialization(self) -> None:
        """
        Test serialization
        """
        fasttext = load_facebook_model("tests/resources/fasttext.bin")
        emb = FasttextNgramEmbeddingSequence(split_tokenizer, fasttext,
                                             sequence_maxlen=10, normalized=True)
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

        self.assertEqual(preproc_embs[1], preproc_embs2[1])
        self.assertEqual(preproc_embs[2], preproc_embs2[2])
        self.assertEqual(preproc_embs[3], preproc_embs2[3])
        self.assertEqual(preproc_embs[4], preproc_embs2[4])

    def test_spam_train(self) -> None:
        """
        Test spam train
        """
        set_seed(42)

        fasttext = load_facebook_model("tests/resources/fasttext.bin")
        emb = FasttextNgramEmbeddingSequence(split_tokenizer, fasttext,
                                             sequence_maxlen=10, normalized=True)

        texts, labels = read_sms_spam_dset()

        model = Sequential(
            [
                Input(emb.outputs_shape.shape),
                Masking(0.0),
                GRU(10, activation="tanh"),
                Dense(10, activation="tanh"),
                Dense(1, activation="sigmoid")
            ]
        )
        model = KerasWrapper(emb, model)
        model.build(texts)
        print(model)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        hst = model.fit(texts, np.asarray(labels), batch_size=256, epochs=6, validation_split=0.1)
        self.assertAlmostEqual(hst.history["accuracy"][-1], 1.0, delta=0.2)
        self.assertAlmostEqual(hst.history["val_accuracy"][-1], 1.0, delta=0.2)
        for i in range(len(hst.history["loss"]) - 1):
            self.assertLess(hst.history["loss"][i + 1], hst.history["loss"][i])
