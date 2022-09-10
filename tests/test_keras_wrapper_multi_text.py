"""
Keras wrapper multi-text tests
"""

import unittest

import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, GRU, Dense, Subtract, Masking, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed

from gianlp.models import KerasWrapper, CharEmbeddingSequence, BaseModel
from tests.utils import LOREM_IPSUM


class TestKerasWrapperMultiTexts(unittest.TestCase):
    """
    Keras wrapper multi-text tests
    """

    def test_simple_concatenate(self) -> None:
        """
        Test concatenation
        """
        char_emb1 = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)
        char_emb2 = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential([Input((32,)), Dense(5, activation="tanh")])
        model = KerasWrapper([("text1", [char_emb1]), ("text2", [char_emb2])], model)
        model.build(LOREM_IPSUM.split("\n"))
        self.assertEqual(model.outputs_shape.shape, (10, 5))

    def test_simple_siamese(self) -> None:
        """
        Test a simple siamese NN
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(10, activation="tanh")]
        )
        encoder = KerasWrapper(char_emb, model)

        siamese = Sequential([Input((20,)), Dense(1, activation="sigmoid")])
        siamese = KerasWrapper([("text1", [encoder]), ("text2", [encoder])], siamese)
        siamese.build(LOREM_IPSUM.split("\n"))
        self.assertEqual(siamese.outputs_shape.shape, (1,))

    def test_multi_input_siamese(self) -> None:
        """
        Test a multi input siamese NN
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(10, activation="tanh")]
        )
        encoder = KerasWrapper(char_emb, model)

        inp1 = Input((10,))
        inp2 = Input((10,))
        subs = Subtract()([inp1, inp2])
        out = Dense(1, activation="sigmoid")(subs)
        siamese = Model(inputs=[inp1, inp2], outputs=out)

        siamese = KerasWrapper([("text1", [encoder]), ("text2", [encoder])], siamese)
        siamese.build(LOREM_IPSUM.split("\n"))
        self.assertEqual(siamese.outputs_shape.shape, (1,))

    def test_multi_input_preprocess_and_predict_consistent(self) -> None:
        """
        Test that the preprocessed texts have the same predictions than the ones with predict
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(10, activation="tanh")]
        )
        encoder = KerasWrapper(char_emb, model)

        inp1 = Input((10,))
        inp2 = Input((10,))
        subs = Subtract()([inp1, inp2])
        out = Dense(1, activation="sigmoid")(subs)
        siamese = Model(inputs=[inp1, inp2], outputs=out)

        siamese = KerasWrapper([("text1", [encoder]), ("text2", [encoder])], siamese)
        siamese.build(LOREM_IPSUM.split("\n"))

        preds1 = siamese.predict({"text1": ["asd", "fgh"], "text2": ["fgh", "asd"]})
        preds2 = siamese(siamese.preprocess_texts({"text1": ["asd", "fgh"], "text2": ["fgh", "asd"]}))

        self.assertEqual(preds1.tolist(), preds2.tolist())

    def test_multi_input_preprocess_and_predict_consistent_df(self) -> None:
        """
        Test that the preprocessed texts have the same predictions than the ones with predict using dataframes
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(10, activation="tanh")]
        )
        encoder = KerasWrapper(char_emb, model)

        inp1 = Input((10,))
        inp2 = Input((10,))
        subs = Subtract()([inp1, inp2])
        out = Dense(1, activation="sigmoid")(subs)
        siamese = Model(inputs=[inp1, inp2], outputs=out)

        siamese = KerasWrapper([("text1", [encoder]), ("text2", [encoder])], siamese)
        siamese.build(LOREM_IPSUM.split("\n"))

        df = pd.DataFrame.from_dict({"text1": ["asd", "fgh"], "text2": ["fgh", "asd"]})

        preds1 = siamese.predict(df)
        preds2 = siamese(siamese.preprocess_texts(df))

        self.assertEqual(preds1.tolist(), preds2.tolist())

    def test_multi_input_preprocess_and_predict_consistent_multi_output(self) -> None:
        """
        Test that the preprocessed texts have the same predictions than the ones with predict with multiple outputs
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(10, activation="tanh")]
        )
        encoder = KerasWrapper(char_emb, model)

        inp1 = Input((10,))
        inp2 = Input((10,))
        subs = Subtract()([inp1, inp2])
        out = Dense(1, activation="sigmoid")(subs)
        out2 = Dense(1, activation="sigmoid")(subs)
        siamese = Model(inputs=[inp1, inp2], outputs=[out, out2])

        siamese = KerasWrapper([("text1", [encoder]), ("text2", [encoder])], siamese)
        siamese.build(LOREM_IPSUM.split("\n"))

        df = pd.DataFrame.from_dict({"text1": ["asd", "fgh"], "text2": ["fgh", "asd"]})

        preds1 = siamese.predict(df)
        preds2 = siamese(siamese.preprocess_texts(df))

        self.assertEqual(
            [p.round(3).astype("float32").tolist() for p in preds1],
            [p.round(3).astype("float32").tolist() for p in preds2],
        )

    def test_train_with_dicts(self) -> None:
        """
        Test for training with dicts
        """
        set_seed(42)
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [
                Input(char_emb.outputs_shape.shape),
                Masking(0.0),
                GRU(10, activation="tanh"),
                Dense(10, activation="tanh"),
            ]
        )
        encoder = KerasWrapper(char_emb, model)

        inp1 = Input((10,))
        inp2 = Input((10,))
        subs = Concatenate()([inp1, inp2])
        x = Dense(10, activation="tanh")(subs)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        out = Dense(1, activation="sigmoid")(x)
        siamese = Model(inputs=[inp1, inp2], outputs=out)

        siamese = KerasWrapper([("text1", [encoder]), ("text2", [encoder])], siamese)
        siamese.build(LOREM_IPSUM.split("\n"))
        print(siamese)
        siamese.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
        hst = siamese.fit(
            {"text1": ["asd", "fgh"] * 2 * 10, "text2": ["asd", "asd", "fgh", "fgh"] * 10},
            np.asarray([0, 1, 1, 0] * 10),
            batch_size=7,
            epochs=20,
        )
        self.assertAlmostEqual(hst.history["accuracy"][-1], 1, delta=0.01)

        preds1 = siamese.predict({"text1": ["asd", "fgh"], "text2": ["asd", "asd"]})
        self.assertEqual([1 if p > 0.5 else 0 for p in preds1], [0, 1])

        # Serialization works after training
        preds1 = siamese.predict({"text1": ["asd", "fgh"], "text2": ["asd", "asd"]})
        serialized = siamese.serialize()
        siamese: KerasWrapper = BaseModel.deserialize(serialized)
        preds2 = siamese.predict({"text1": ["asd", "fgh"], "text2": ["asd", "asd"]})
        self.assertEqual(preds1.tolist(), preds2.tolist())

        # Training does corrupt encoder behaviour
        preds1 = encoder.predict(["asd", "fgh"])
        serialized = encoder.serialize()
        encoder: KerasWrapper = BaseModel.deserialize(serialized)
        preds2 = encoder.predict(["asd", "fgh"])
        self.assertEqual(preds1.tolist(), preds2.tolist())

    def test_train_with_dataframe(self) -> None:
        """
        Test for training with dataframe
        """
        set_seed(42)
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [
                Input(char_emb.outputs_shape.shape),
                Masking(0.0),
                GRU(10, activation="tanh"),
                Dense(10, activation="tanh"),
            ]
        )
        encoder = KerasWrapper(char_emb, model)

        inp1 = Input((10,))
        inp2 = Input((10,))
        subs = Concatenate()([inp1, inp2])
        x = Dense(10, activation="tanh")(subs)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        out = Dense(1, activation="sigmoid")(x)
        siamese = Model(inputs=[inp1, inp2], outputs=out)

        siamese = KerasWrapper([("text1", [encoder]), ("text2", [encoder])], siamese)
        siamese.build(LOREM_IPSUM.split("\n"))
        siamese.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
        hst = siamese.fit(
            pd.DataFrame.from_dict({"text1": ["asd", "fgh"] * 2 * 10, "text2": ["asd", "asd", "fgh", "fgh"] * 10}),
            np.asarray([0, 1, 1, 0] * 10),
            batch_size=7,
            epochs=20,
        )
        self.assertAlmostEqual(hst.history["accuracy"][-1], 1, delta=0.01)

        preds1 = siamese.predict(pd.DataFrame.from_dict({"text1": ["asd", "fgh"], "text2": ["asd", "asd"]}))
        self.assertEqual([1 if p > 0.5 else 0 for p in preds1], [0, 1])

    def test_train_validation_split(self) -> None:
        """
        Test for training with validation split
        """
        set_seed(42)
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [
                Input(char_emb.outputs_shape.shape),
                Masking(0.0),
                GRU(10, activation="tanh"),
                Dense(10, activation="tanh"),
            ]
        )
        encoder = KerasWrapper(char_emb, model)

        inp1 = Input((10,))
        inp2 = Input((10,))
        subs = Concatenate()([inp1, inp2])
        x = Dense(10, activation="tanh")(subs)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        out = Dense(1, activation="sigmoid")(x)
        siamese = Model(inputs=[inp1, inp2], outputs=out)

        siamese = KerasWrapper([("text1", [encoder]), ("text2", [encoder])], siamese)
        siamese.build(LOREM_IPSUM.split("\n"))
        siamese.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
        hst = siamese.fit(
            {"text1": ["asd", "fgh"] * 2 * 20, "text2": ["asd", "asd", "fgh", "fgh"] * 20},
            np.asarray([0, 1, 1, 0] * 20),
            batch_size=7,
            epochs=20,
            validation_split=0.1,
        )
        self.assertAlmostEqual(hst.history["accuracy"][-1], 1, delta=0.01)
        self.assertAlmostEqual(hst.history["val_accuracy"][-1], 1, delta=0.01)

    def test_train_validation_split_dataframe(self) -> None:
        """
        Test for training with a dataframe and validation split
        """
        set_seed(42)
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [
                Input(char_emb.outputs_shape.shape),
                Masking(0.0),
                GRU(10, activation="tanh"),
                Dense(10, activation="tanh"),
            ]
        )
        encoder = KerasWrapper(char_emb, model)

        inp1 = Input((10,))
        inp2 = Input((10,))
        subs = Concatenate()([inp1, inp2])
        x = Dense(10, activation="tanh")(subs)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        out = Dense(1, activation="sigmoid")(x)
        siamese = Model(inputs=[inp1, inp2], outputs=out)

        siamese = KerasWrapper([("text1", [encoder]), ("text2", [encoder])], siamese)
        siamese.build(LOREM_IPSUM.split("\n"))
        siamese.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
        hst = siamese.fit(
            pd.DataFrame.from_dict({"text1": ["asd", "fgh"] * 2 * 20, "text2": ["asd", "asd", "fgh", "fgh"] * 20}),
            np.asarray([0, 1, 1, 0] * 20),
            batch_size=7,
            epochs=20,
            validation_split=0.1,
        )
        self.assertAlmostEqual(hst.history["accuracy"][-1], 1, delta=0.01)
        self.assertAlmostEqual(hst.history["val_accuracy"][-1], 1, delta=0.01)

    def test_serialization(self) -> None:
        """
        Test for training with dicts
        """
        set_seed(42)
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [
                Input(char_emb.outputs_shape.shape),
                Masking(0.0),
                GRU(10, activation="tanh"),
                Dense(10, activation="tanh"),
            ]
        )
        encoder = KerasWrapper(char_emb, model)

        inp1 = Input((10,))
        inp2 = Input((10,))
        subs = Concatenate()([inp1, inp2])
        x = Dense(10, activation="tanh")(subs)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        out = Dense(1, activation="sigmoid")(x)
        siamese = Model(inputs=[inp1, inp2], outputs=out)

        siamese = KerasWrapper([("text1", [encoder]), ("text2", [encoder])], siamese)
        siamese.build(LOREM_IPSUM.split("\n"))
        siamese.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
        siamese.fit(
            {"text1": ["asd", "fgh"] * 2 * 10, "text2": ["asd", "asd", "fgh", "fgh"] * 10},
            np.asarray([0, 1, 1, 0] * 10),
            batch_size=7,
            epochs=2,
        )

        preds1 = siamese.predict({"text1": ["asd", "fgh"], "text2": ["asd", "asd"]})
        serialized = siamese.serialize()
        siamese: KerasWrapper = BaseModel.deserialize(serialized)
        preds2 = siamese.predict({"text1": ["asd", "fgh"], "text2": ["asd", "asd"]})
        self.assertEqual(preds1.tolist(), preds2.tolist())

        # Serialized objects can be serialized again
        preds1 = siamese.predict({"text1": ["asd", "fgh"], "text2": ["asd", "asd"]})
        serialized = siamese.serialize()
        siamese: KerasWrapper = BaseModel.deserialize(serialized)
        preds2 = siamese.predict({"text1": ["asd", "fgh"], "text2": ["asd", "asd"]})
        self.assertEqual(preds1.tolist(), preds2.tolist())

    def test_freeze(self) -> None:
        """
        Test for training with dicts
        """
        set_seed(42)
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [
                Input(char_emb.outputs_shape.shape),
                Masking(0.0),
                GRU(10, activation="tanh"),
                Dense(10, activation="tanh"),
            ]
        )
        encoder = KerasWrapper(char_emb, model)

        inp1 = Input((10,))
        inp2 = Input((10,))
        subs = Concatenate()([inp1, inp2])
        x = Dense(10, activation="tanh")(subs)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        x = Dense(10, activation="tanh")(x)
        out = Dense(1, activation="sigmoid")(x)
        siamese = Model(inputs=[inp1, inp2], outputs=out)

        siamese = KerasWrapper([("text1", [encoder]), ("text2", [encoder])], siamese)
        siamese.build(LOREM_IPSUM.split("\n"))

        siamese.freeze()

        self.assertEqual(siamese.trainable_weights_amount, 0)

        preds1 = siamese.predict({"text1": ["asd", "fgh"], "text2": ["asd", "asd"]})
        serialized = siamese.serialize()
        siamese: KerasWrapper = BaseModel.deserialize(serialized)
        preds2 = siamese.predict({"text1": ["asd", "fgh"], "text2": ["asd", "asd"]})
        self.assertEqual(preds1.tolist(), preds2.tolist())

        # still frozen
        self.assertEqual(siamese.trainable_weights_amount, 0)

    def test_simple_multi_text_build(self) -> None:
        """
        Test simple multi-text build
        """
        char_emb1 = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)
        char_emb2 = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential([Input((32,)), Dense(5, activation="tanh")])
        model = KerasWrapper([("text1", [char_emb1]), ("text2", [char_emb2])], model)
        model.build({"text1": LOREM_IPSUM.split("\n"), "text2": LOREM_IPSUM.split("\n")})
        self.assertEqual(model.outputs_shape.shape, (10, 5))

    def test_multi_text_build_bifurcation(self) -> None:
        """
        Test multi-text build with texts for each input
        """
        char_emb1 = CharEmbeddingSequence(
            embedding_dimension=16, sequence_maxlen=1, min_freq_percentile=0, random_state=42
        )
        char_emb2 = CharEmbeddingSequence(
            embedding_dimension=16, sequence_maxlen=1, min_freq_percentile=0, random_state=42
        )

        model = Sequential([Input((32,)), Dense(5, activation="tanh")])
        model = KerasWrapper([("text1", [char_emb1]), ("text2", [char_emb2])], model)
        model.build({"text1": ["asd", "dsa"], "text2": ["qwe", "ewq"]})

        self.assertEqual(
            char_emb1(char_emb1.preprocess_texts(["q"])).tolist(), char_emb2(char_emb2.preprocess_texts(["a"])).tolist()
        )

        self.assertEqual(
            char_emb1(char_emb1.preprocess_texts(["q"])).tolist(), char_emb2(char_emb2.preprocess_texts(["s"])).tolist()
        )

        self.assertEqual(
            char_emb1(char_emb1.preprocess_texts(["a"])).tolist(), char_emb2(char_emb2.preprocess_texts(["q"])).tolist()
        )

        self.assertEqual(model.outputs_shape.shape, (1, 5))

    def test_multi_text_build_with_extra_key(self) -> None:
        """
        Test multi-text build with an extra key. Extra key gets ignored.
        """
        char_emb1 = CharEmbeddingSequence(
            embedding_dimension=16, sequence_maxlen=1, min_freq_percentile=0, random_state=42
        )
        char_emb2 = CharEmbeddingSequence(
            embedding_dimension=16, sequence_maxlen=1, min_freq_percentile=0, random_state=42
        )

        model = Sequential([Input((32,)), Dense(5, activation="tanh")])
        model = KerasWrapper([("text1", [char_emb1]), ("text2", [char_emb2])], model)
        model.build({"text1": ["asd", "dsa"], "text2": ["qwe", "ewq"], "text3": ["322", "434"]})

        self.assertEqual(
            char_emb1(char_emb1.preprocess_texts(["q"])).tolist(), char_emb2(char_emb2.preprocess_texts(["a"])).tolist()
        )

        self.assertEqual(
            char_emb1(char_emb1.preprocess_texts(["q"])).tolist(), char_emb2(char_emb2.preprocess_texts(["s"])).tolist()
        )

        self.assertEqual(
            char_emb1(char_emb1.preprocess_texts(["a"])).tolist(), char_emb2(char_emb2.preprocess_texts(["q"])).tolist()
        )

        self.assertEqual(model.outputs_shape.shape, (1, 5))

    def test_multi_text_build_with_missing_key_raises_error(self) -> None:
        """
        Test multi-text build with a missing key raises a value error.
        """
        char_emb1 = CharEmbeddingSequence(
            embedding_dimension=16, sequence_maxlen=1, min_freq_percentile=0, random_state=42
        )
        char_emb2 = CharEmbeddingSequence(
            embedding_dimension=16, sequence_maxlen=1, min_freq_percentile=0, random_state=42
        )

        model = Sequential([Input((32,)), Dense(5, activation="tanh")])
        model = KerasWrapper([("text1", [char_emb1]), ("text2", [char_emb2])], model)
        with self.assertRaises(ValueError):
            model.build({"text1": ["asd", "dsa"]})
