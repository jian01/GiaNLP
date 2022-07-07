"""
Masked embedding class module
"""

import tensorflow as tf

# pylint: disable=no-name-in-module
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Embedding
# pylint: enable=no-name-in-module


@tf.keras.utils.register_keras_serializable()
class MaskedEmbedding(Embedding):
    """
    Special class for a masked at 0 embedding.

    It guarantees that the index 0 always maps to a vector of zeros.
    """

    def __init__(self, input_dim, output_dim, **kwargs):
        """

        :param input_dim:Size of the vocabulary,
        i.e. maximum integer index + 1.
        :param output_dim: Integer. Dimension of the dense embedding.
        :param kwargs: extra arguments to pass to Embedding init
        """
        super().__init__(input_dim, output_dim, **kwargs)

    def call(self, inputs):
        """
        Wraps the original call for guaranteeing masking
        :param inputs: inputs to forward the layer
        :return: output of the forward pass
        """
        out = super().call(inputs)

        dim_to_expand = out.shape.ndims - 1

        inputs_mask = K.repeat_elements(
            tf.cast(tf.expand_dims(inputs != 0, dim_to_expand), out.dtype), rep=self.output_dim, axis=dim_to_expand
        )
        return tf.multiply(out, inputs_mask)
