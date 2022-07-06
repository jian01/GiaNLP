"""
Module for RNN Digest model
"""
from typing import Union

from tensorflow.keras.layers import Input, GRU, LSTM, Masking, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.random import set_seed

from gianlp.models.base_model import BaseModel, ModelInputs
from gianlp.models.keras_wrapper import KerasWrapper


class RNNDigest(KerasWrapper):
    """
    Model for RNN digest, independent of sequence length and ndims
    """

    def __new__(
        cls,
        inputs: Union[ModelInputs, BaseModel],
        units_per_layer: int,
        rnn_type: str,
        stacked_layers: int = 1,
        masking: bool = True,
        bidirectional: bool = False,
        return_sequences: bool = False,
        random_seed: int = 42,
        **kwargs
    ):
        """

        :param inputs: the inputs of the model
        :param units_per_layer: the amount of units per layer
        :param rnn_type: the type of rnn, could be "gru" or "lstm"
        :param stacked_layers: the amount of layers to stack, 1 by default
        :param masking: if apply masking with 0 to the sequence
        :param bidirectional: if it's bidirectional
        :param random_seed: the seed for random processes
        :param return_sequences: if True, the last RNN layer returns the sequence of states
        :param kwargs: extra arguments for the rnn layers
        :raises:
            ValueError: When inputs have different sequence length
        """
        set_seed(random_seed)

        if isinstance(inputs, list):
            iterator = cls._iterate_model_inputs(inputs)
            initial_shape = next(iterator).outputs_shape.shape[-2:]
            for inp in iterator:
                if inp.outputs_shape.shape[-2] != initial_shape[0]:
                    raise ValueError("Inputs have different sequence length")
                initial_shape = (initial_shape[0], initial_shape[1] + inp.outputs_shape.shape[-1])
        else:
            initial_shape = inputs.outputs_shape.shape[-2:]

        model = Sequential([Input(initial_shape)])
        if masking:
            model.add(Masking(0.0))
        layer_to_use = LSTM
        if rnn_type == "gru":
            layer_to_use = GRU
        for _ in range(stacked_layers - 1):
            if bidirectional:
                model.add(Bidirectional(layer_to_use(units_per_layer, return_sequences=True, **kwargs)))
            else:
                model.add(layer_to_use(units_per_layer, return_sequences=True, **kwargs))
        if bidirectional:
            model.add(Bidirectional(layer_to_use(units_per_layer, **kwargs)))
        else:
            model.add(layer_to_use(units_per_layer, return_sequences=return_sequences, **kwargs))

        return KerasWrapper(inputs, model, random_seed=random_seed)
