"""
Keras wrapper exceptions tests
"""

import unittest

from tensorflow.keras.layers import Input, GRU, Dense, Masking, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.models import Sequential, Model

from gianlp.models import KerasWrapper, CharEmbeddingSequence, PerChunkSequencer
from tests.utils import dot_chunker, LOREM_IPSUM


class TestKerasWrapperExceptions(unittest.TestCase):
    """
    Keras wrapper exceptions tests
    """

    def test_no_input_exception(self) -> None:
        """
        Test raises value error when the model has no input
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential([Dense(5, activation="tanh")])
        with self.assertRaises(ValueError):
            KerasWrapper(char_emb, model)

    def test_wraped_not_keras_exception(self) -> None:
        """
        Test raises value error when the wrapped model is not a keras model
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential([Input(char_emb.outputs_shape.shape), Dense(5, activation="tanh")])
        model = KerasWrapper(char_emb, model)
        with self.assertRaises(ValueError):
            KerasWrapper(model, model)

    def test_call_not_built_model(self) -> None:
        """
        Test raises value error when the model was not built
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential([Input(char_emb.outputs_shape.shape), Dense(5, activation="tanh")])
        model = KerasWrapper(char_emb, model)
        with self.assertRaises(ValueError):
            model(Input((10,)))

    def test_mixed_named_inputs_with_not_named(self) -> None:
        """
        Test raises value error when the inputs are a dict of length 1
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential([Input(char_emb.outputs_shape.shape), Dense(5, activation="tanh")])
        with self.assertRaises(ValueError):
            KerasWrapper([char_emb, ("title", [char_emb])], model)

    def test_multi_text_input_length_1(self) -> None:
        """
        Test raises value error when the inputs are a dict of length 1
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential([Input(char_emb.outputs_shape.shape), Dense(5, activation="tanh")])
        with self.assertRaises(ValueError):
            KerasWrapper([("title", [char_emb])], model)

    def test_multi_text_list_inconsistence(self) -> None:
        """
        Test exception when inputs is a list and some of the models in the input have multi-text input and others don't.
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(10, activation="tanh")]
        )
        encoder = KerasWrapper(char_emb, model)

        encoder_title_content = Sequential([Input((20,)), Dense(20, activation="tanh")])
        encoder_title_content = KerasWrapper([("title", [encoder]), ("content", [encoder])], encoder_title_content)
        encoder_description = Sequential([Input((10,)), Dense(10, activation="tanh")])
        encoder_description = KerasWrapper(encoder, encoder_description)

        final_model = Sequential([Input((30,)), Dense(1, activation="sigmoid")])
        with self.assertRaises(ValueError):
            KerasWrapper([encoder_title_content, encoder_description], final_model)

    def test_multi_text_dict_inconsistence(self) -> None:
        """
        Test exception when inputs is a dict and any of the models has multi-text input.
        """
        char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=10)

        model = Sequential(
            [Input(char_emb.outputs_shape.shape), GRU(10, activation="tanh"), Dense(10, activation="tanh")]
        )
        encoder = KerasWrapper(char_emb, model)

        encoder_title_content = Sequential([Input((20,)), Dense(20, activation="tanh")])
        encoder_title_content = KerasWrapper([("title", [encoder]), ("content", [encoder])], encoder_title_content)
        encoder_description = Sequential([Input((10,)), Dense(10, activation="tanh")])
        encoder_description = KerasWrapper(encoder, encoder_description)

        final_model = Sequential([Input((30,)), Dense(1, activation="sigmoid")])
        with self.assertRaises(ValueError):
            KerasWrapper([("title", [encoder_title_content]), ("description", [encoder_description])], final_model)

    def test_inference_multi_text_errors(self) -> None:
        """
        Test invalid text format while inferencing
        """
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
        siamese.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        with self.assertRaises(ValueError):
            siamese.predict(["asd", "asd"])
        with self.assertRaises(ValueError):
            encoder.predict({"text1": ["asd"], "text2": ["asd"]})
