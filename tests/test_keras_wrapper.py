"""
Keras wrapper utils
"""

import random
import string
import unittest

import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, GRU, Dense, Subtract, Masking, Conv1D, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

from gianlp.models import KerasWrapper, BaseModel, CharEmbeddingSequence
from tests.utils import LOREM_IPSUM, accuracy, generator_from_list, set_seed


class TestKerasWrapper(unittest.TestCase):
    """
    Keras wrapper utils
    """

    def test_simple_gru_classifier_shapes(self) -> None:
        """
        Test with a simple GRU classifier shapes
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(char_emb, model)
        self.assertEqual(model.inputs_shape.shape, (10,))
        self.assertEqual(model.outputs_shape.shape, (1,))
        model.build(LOREM_IPSUM.split(" "))
        self.assertEqual(model.outputs_shape.shape, (1,))
        print(model)

    def test_preprocess_forward(self) -> None:
        """
        Test that preprocessing and forwarding gives the same result as predicting
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(char_emb, model)
        model.build(LOREM_IPSUM.split(" "))
        preds1 = model.predict(pd.Series(["asd", "fgh"]))
        preprocessed = model.preprocess_texts(pd.Series(["asd", "fgh"]))
        preds2 = model(preprocessed)
        self.assertEqual(preds1.tolist(), preds2.tolist())

    def test_simple_input_formats(self) -> None:
        """
        Test simple input list and Series formats with a simple GRU classifier
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(char_emb, model)
        self.assertEqual(model.inputs_shape.shape, (10,))
        model.build(pd.Series(LOREM_IPSUM.split("\n")))
        preds1 = model.predict(["asd", "fgh"])
        preds2 = model.predict(pd.Series(["asd", "fgh"]))
        self.assertEqual(preds1.tolist(), preds2.tolist())
        print(model)

    def test_predict_inference_batches(self) -> None:
        """
        Test inference batch sizes
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(char_emb, model)
        self.assertEqual(model.inputs_shape.shape, (10,))
        model.build(pd.Series(LOREM_IPSUM.split("\n")))
        preds1 = model.predict(["asd", "fgh", "123", "jjj"], inference_batch=1)
        preds2 = model.predict(["asd", "fgh", "123", "jjj"], inference_batch=2)
        preds3 = model.predict(["asd", "fgh", "123", "jjj"], inference_batch=3)
        self.assertEqual(preds1.tolist(), preds2.tolist())
        self.assertEqual(preds1.tolist(), preds3.tolist())

    def test_simple_dense_shapes(self) -> None:
        """
        Test with a simple dense modifying each embedding shapes
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential([Input((16,)), Dense(5, activation="tanh")])
        model = KerasWrapper(char_emb, model)
        self.assertEqual(model.inputs_shape.shape, (10,))
        self.assertEqual(model.outputs_shape.shape, (10, 5))
        model.build(LOREM_IPSUM.split(" "))
        self.assertEqual(model.outputs_shape.shape, (10, 5))
        print(model)

    def test_simple_concatenate_shapes(self) -> None:
        """
        Test concatenation shapes with simple input
        """
        char_emb1 = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)
        char_emb2 = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential([Input((32,)), Dense(5, activation="tanh")])
        model = KerasWrapper([char_emb1, char_emb2], model)
        self.assertEqual([s.shape for s in model.inputs_shape], [(10,), (10,)])
        self.assertEqual(model.outputs_shape.shape, (10, 5))
        model.build(LOREM_IPSUM.split(" "))
        self.assertEqual(model.outputs_shape.shape, (10, 5))
        print(model)

    def test_simple_siamese_shapes(self) -> None:
        """
        Test a simple siamese NN shapes
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(15, activation="tanh")]
        )
        encoder = KerasWrapper(char_emb, model)

        siamese = Sequential([Input((30,)), Dense(1, activation="sigmoid")])
        siamese = KerasWrapper([encoder, encoder], siamese)
        self.assertEqual([s.shape for s in siamese.inputs_shape], [(10,), (10,)])
        self.assertEqual(encoder.inputs_shape.shape, (10,))
        self.assertEqual(encoder.outputs_shape.shape, (15,))
        self.assertEqual(siamese.outputs_shape.shape, (1,))
        siamese.build(LOREM_IPSUM.split(" "))
        self.assertEqual(encoder.outputs_shape.shape, (15,))
        self.assertEqual(siamese.outputs_shape.shape, (1,))
        print(siamese)

    def test_multi_input_siamese_shapes(self) -> None:
        """
        Test a multi input siamese NN shapes
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(15, activation="tanh")]
        )
        encoder = KerasWrapper(char_emb, model)

        inp1 = Input(encoder.outputs_shape.shape)
        inp2 = Input(encoder.outputs_shape.shape)
        subs = Subtract()([inp1, inp2])
        out = Dense(1, activation="sigmoid")(subs)
        siamese = Model(inputs=[inp1, inp2], outputs=out)

        siamese = KerasWrapper([encoder, encoder], siamese)
        self.assertEqual([s.shape for s in siamese.inputs_shape], [(10,), (10,)])
        self.assertEqual(encoder.inputs_shape.shape, (10,))
        self.assertEqual(encoder.outputs_shape.shape, (15,))
        self.assertEqual(siamese.outputs_shape.shape, (1,))
        siamese.build(LOREM_IPSUM.split(" "))
        self.assertEqual(encoder.outputs_shape.shape, (15,))
        self.assertEqual(siamese.outputs_shape.shape, (1,))
        print(siamese)

    def test_multiple_compile(self) -> None:
        """
        Compile multiple models with the same inputs
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        encoder = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(10, activation="tanh")]
        )
        encoder = KerasWrapper(char_emb, encoder)

        classifier = Sequential([Input((10,)), Dense(1, activation="sigmoid")])
        classifier = KerasWrapper(encoder, classifier)

        regressor = Sequential([Input((10,)), Dense(1, activation="relu")])
        regressor = KerasWrapper(encoder, regressor)

        self.assertEqual(classifier.outputs_shape.shape, (1,))
        self.assertEqual(regressor.outputs_shape.shape, (1,))

        classifier.build(LOREM_IPSUM.split(" "))
        regressor.build(LOREM_IPSUM.split(" "))

        self.assertEqual(classifier.outputs_shape.shape, (1,))
        self.assertEqual(regressor.outputs_shape.shape, (1,))

    def test_gru_classifier_overfit(self) -> None:
        """
        Test fiting a GRU classifier
        """
        set_seed(42)
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(char_emb, model)
        model.build(LOREM_IPSUM.split(" "))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        hst = model.fit(["asd", "bcd"] * 6, np.asarray([0, 1] * 6), batch_size=1, epochs=10)
        self.assertAlmostEqual(hst.history["accuracy"][-1], 1.0, delta=0.01)
        preds = model.predict(["asd", "bcd"] * 6)
        self.assertTrue((preds > 0.5).flatten().tolist() == [False, True] * 6)
        self.assertAlmostEqual(hst.history["accuracy"][-1], 1.0, delta=0.01)

    def test_gru_classifier_underfit(self):
        """
        Test fiting a GRU classifier with a char embedding that makes learning unfeasible
        """
        set_seed(42)
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10, min_freq_percentile=95)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(char_emb, model)
        model.build(LOREM_IPSUM.split(" "))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        hst = model.fit(
            ["asd", "bcd"] * 6,
            np.asarray([0, 1] * 6),
            batch_size=1,
            epochs=10,
            validation_data=(["asd", "bcd"] * 6, np.asarray([0, 1] * 6)),
        )
        self.assertAlmostEqual(hst.history["accuracy"][-1], 0.5, delta=0.01)
        preds = model.predict(["asd", "bcd"] * 6)
        self.assertAlmostEqual(preds.mean(), 0.5, delta=0.01)
        self.assertAlmostEqual(preds.std(), 0, delta=0.01)

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
                    "".join(random.choice(letter_with_no_vocals + string.digits) for _ in range(random.randint(2, 8)))
                ]
            for i in range(64):
                if random.randint(0, 1) == 1:
                    texts[i] = texts[i][1:] + random.choice(vocals)
                    labels.append(1)
                else:

                    labels.append(0)
            yield texts, np.asarray(labels)

    def test_classifier_with_generators(self) -> None:
        """
        Training with generators and predict test
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

        model.build(LOREM_IPSUM.split(" "))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        hst = model.fit(
            self.starts_with_vocal_generator(),
            epochs=30,
            steps_per_epoch=100,
            validation_data=self.starts_with_vocal_generator(),
            validation_steps=10,
        )
        self.assertAlmostEqual(hst.history["accuracy"][-1], 1.0, delta=0.15)
        self.assertAlmostEqual(hst.history["val_accuracy"][-1], 1.0, delta=0.15)
        preds = model.predict(["A", "E", "I", "O", "U", "JS4DS", "S4DS", "4DS", "DS", "S"])
        preds = [1 if p > 0.5 else 0 for p in preds.flatten()]
        self.assertAlmostEqual(accuracy([1] * 5 + [0] * 5, preds), 1.0, delta=0.15)

    def test_fit_with_generators_no_validation(self) -> None:
        """
        Training fiting with generators with no validation data
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

        model.build(LOREM_IPSUM.split(" "))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        hst = model.fit(self.starts_with_vocal_generator(), epochs=30, steps_per_epoch=100)
        self.assertAlmostEqual(hst.history["accuracy"][-1], 1.0, delta=0.15)

    def test_serialization(self) -> None:
        """
        Test serialization with a simple GRU classifier
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(char_emb, model)
        serialized = model.serialize()
        model: KerasWrapper = BaseModel.deserialize(serialized)
        model.build(LOREM_IPSUM.split(" "))
        serialized = model.serialize()
        model2: KerasWrapper = BaseModel.deserialize(serialized)
        self.assertEqual(model.predict(["asd"]).tolist(), model2.predict(["asd"]).tolist())

    def test_train_generator_shuffles(self) -> None:
        """
        Test the shuffler of the train generator

        If the data is never shuffled between epochs the accuracy is never 1 and oscilates always with accuracy 3/4.
        The test is kinda a XOR (inspired by something that failed coding
        TestKerasWrapperMultiTexts.test_train_with_dicts)

        I know this test may seem falopa, if you are experiencing issues with it please think twice before doing
        anything, this was thought and tested carefully.
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
        siamese.build(LOREM_IPSUM.split(" "))
        siamese.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
        hst = siamese.fit(
            {"text1": ["asd", "fgh"] * 2, "text2": ["asd", "asd", "fgh", "fgh"]},
            np.asarray([0, 1, 1, 0]),
            batch_size=3,
            epochs=50,
        )
        one_count = 0
        for acc in hst.history["accuracy"]:
            if acc == 1:
                one_count += 1
        self.assertGreater(one_count, 1)

    def test_predict_with_generator(self) -> None:
        """
        Test predicting with a generator
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(char_emb, model)
        model.build(LOREM_IPSUM.split(" "))
        preds1 = model.predict(["asd", "123", "test"])
        preds2 = model.predict(generator_from_list([["asd", "123"], ["test"]]), steps=3)
        self.assertEqual(preds1.tolist(), preds2.tolist())
        preds2 = model.predict(generator_from_list([["asd", "123"], ["test"]]), steps=20)
        self.assertEqual(preds1.tolist(), preds2.tolist())

    def test_predict_with_generator_no_steps_error(self) -> None:
        """
        Test predicting with a generator
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(char_emb, model)
        model.build(LOREM_IPSUM.split(" "))
        with self.assertRaises(ValueError):
            model.predict(generator_from_list([["asd", "123"], ["test"]]))

    def test_predict_with_generator_w_multiple_outputs(self) -> None:
        """
        Test predicting with a generator for multiple outputs
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        inp1 = Input(char_emb.outputs_shape.shape)
        gru1 = GRU(10, activation="tanh")(inp1)
        out1 = Dense(1, activation="sigmoid")(gru1)
        out2 = Dense(1, activation="sigmoid")(gru1)
        model = Model(inputs=inp1, outputs=[out1, out2])

        model = KerasWrapper(char_emb, model)
        model.build(LOREM_IPSUM.split(" "))
        preds1 = model.predict(["asd", "123", "test"])
        preds2 = model.predict(generator_from_list([["asd", "123"], ["test"]]), steps=3)
        self.assertEqual([p.tolist() for p in preds1], [p.tolist() for p in preds2])
        preds2 = model.predict(generator_from_list([["asd", "123"], ["test"]]), steps=20)
        self.assertEqual([p.tolist() for p in preds1], [p.tolist() for p in preds2])
        preds2 = model.predict(generator_from_list([["asd"], ["123"], ["test"]]), steps=3)
        self.assertEqual([p.tolist() for p in preds1], [p.tolist() for p in preds2])

    def test_multiple_output_chained(self) -> None:
        """
        Test a model with multiple output followed by another one
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        encoder = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(10, activation="tanh")]
        )
        encoder = KerasWrapper(char_emb, encoder)

        inp = Input((10,))
        classifier = Dense(1, activation="sigmoid")(inp)
        regressor = Dense(1, activation=None)(inp)
        double_out = Model(inputs=inp, outputs=[classifier, regressor])
        double_out = KerasWrapper(encoder, double_out)

        self.assertEqual([s.shape for s in double_out.outputs_shape], [(1,), (1,)])

        inp1 = Input((1,))
        inp2 = Input((1,))
        concat = Concatenate()([inp1, inp2])
        classifier = Dense(1, activation="sigmoid")(concat)
        model = Model(inputs=[inp1, inp2], outputs=classifier)
        model = KerasWrapper(double_out, model)

        self.assertEqual(model.outputs_shape.shape, (1,))

        model.build(LOREM_IPSUM.split(" "))

        self.assertEqual([s.shape for s in double_out.outputs_shape], [(1,), (1,)])
        self.assertEqual(model.outputs_shape.shape, (1,))

    def test_exception_freezing_not_built(self) -> None:
        """
        Test exception when freezing a not built model
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(char_emb, model)
        with self.assertRaises(ValueError):
            model.freeze()

    def test_freezing(self) -> None:
        """
        Test freezing
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(1, activation="sigmoid")]
        )
        model = KerasWrapper(char_emb, model)
        model.build(LOREM_IPSUM.split(" "))
        self.assertGreater(model.trainable_weights_amount, 0)
        self.assertGreater(model.weights_amount, 0)
        model.freeze()
        self.assertEqual(model.trainable_weights_amount, 0)
        self.assertGreater(model.weights_amount, 0)
