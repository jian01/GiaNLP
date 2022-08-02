"""
Module for Keras wrapper model
"""
import pickle
from queue import Queue
from typing import List, Union, Optional, Iterator, cast

# pylint: disable=no-name-in-module
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model

# pylint: enable=no-name-in-module

from gianlp.logging import warning
from gianlp.models.base_model import BaseModel, ModelInputs, ModelIOShape
from gianlp.models.text_representations.text_representation import TextRepresentation
from gianlp.models.trainable_model import TrainableModel
from gianlp.types import SimpleTypeTexts


class KerasWrapper(TrainableModel):
    """
    Keras Model wrapper class.

    Encapsulates a Keras model for using it as a TrainableModel

    :var _inputs: model input models
    :var _wrapped_model: the user wrapped model. should have a defined input shape
    :var _keras_model: the internal keras model
    """

    _inputs: ModelInputs
    _wrapped_model: Model
    _keras_model: Optional[Model]

    @staticmethod
    def __validated_inputs(inputs: Union[ModelInputs, BaseModel]):
        if isinstance(inputs, list) and isinstance(inputs[0], BaseModel):
            for model in inputs[1:]:
                if isinstance(model, tuple):
                    raise ValueError("Can't mix named inputs with unnamed inputs in the list.")
                if model.has_multi_text_input() != inputs[0].has_multi_text_input():
                    raise ValueError(
                        "Some models in the input list have multi-text input and others don't. " "This is not allowed."
                    )
            return inputs
        if isinstance(inputs, list) and isinstance(inputs[0], tuple):
            if len(inputs) == 1:
                raise ValueError("Multi-text input should be used with at least two types of texts.")
            for _, models in inputs:
                for model in models:
                    if model.has_multi_text_input():
                        raise ValueError("Some models in the input dict have multi-text input. This is not allowed.")
            return inputs
        return [inputs]

    def __init__(self, inputs: Union[ModelInputs, BaseModel], wrapped_model: Model, **kwargs):
        """
        :param inputs: the models that are the input of this one. Either a list containing model inputs one by one or a
            dict indicating which text name is assigned to which inputs.
            If a list, all should have multi-text input or don't have it. If it's a dict all shouldn't have multi-text
            input.
        :param wrapped_model: the keras model to wrap. if it has multiple inputs, inputs parameter
            should have the same len
        :param **kwargs: extra parameters for TrainableModel init
        :raises ValueError:
            * When the wrapped model is not a keras model
            * When the keras model to wrap does not have a defined input shape
            * When inputs is a list of models and some of the models in the input have multi-text input and others\
            don't.
            * When inputs is a list of tuples and any of the models has multi-text input.
            * When inputs is a list of tuples with length one
            * When inputs is a list containing some tuples of (str, model) and some models
            * When the wrapped model has multiple inputs and the inputs don't have the same length as the inputs in\
            wrapped model
        """
        if not isinstance(wrapped_model, Model):
            raise ValueError("The model to wrap should be a plain keras model.")

        if not wrapped_model.inputs:
            raise ValueError("The keras model to be wrapped should have a defined input.")

        self._inputs = self.__validated_inputs(inputs)

        self._wrapped_model = wrapped_model
        self._keras_model = None
        super().__init__(**kwargs)

    @property
    def inputs(self) -> ModelInputs:
        """
        Method for getting all models that serve as input

        :return: a list or list of tuples containing BaseModel objects
        """
        return self._inputs

    @property
    def inputs_shape(self) -> Union[List[ModelIOShape], ModelIOShape]:
        """
        Returns the shapes of the inputs of the model

        :return: a list of shape tuple or shape tuple
        """
        shapes: List[ModelIOShape] = []
        for inp in self._iterate_model_inputs(self.inputs):
            shapes += inp.inputs_shape if isinstance(inp.inputs_shape, list) else [inp.inputs_shape]
        if len(shapes) == 1:
            return shapes[0]
        return shapes

    @staticmethod
    def __compute_output_shape(
        output: Tensor, wrapped_model: Model, input_iterator: Iterator[BaseModel]
    ) -> ModelIOShape:
        """
        Compute the output shape for a single output of the wrapped model

        :param output: the single output
        :param wrapped_model: the wrapped model
        :param input_iterator: an iterator of model inputs
        :return: a model shape
        """
        wrapped_output = tuple(output.shape[1:])
        inputs_out_shapes = []
        for inp in input_iterator:
            if isinstance(inp.outputs_shape, list):
                inputs_out_shapes += inp.outputs_shape
            else:
                inputs_out_shapes.append(inp.outputs_shape)

        if len(wrapped_model.inputs) > 1:
            return ModelIOShape(wrapped_output, output.dtype)

        wrapped_initial_input = tuple(wrapped_model.inputs[0].shape[1:])

        if len(inputs_out_shapes[0].shape) > len(wrapped_initial_input):  # TimeDistributed case
            return ModelIOShape(
                inputs_out_shapes[0].shape[: len(inputs_out_shapes[0].shape) - len(wrapped_initial_input)]
                + wrapped_output,
                output.dtype,
            )
        return ModelIOShape(wrapped_output, output.dtype)

    @property
    def outputs_shape(self) -> Union[List[ModelIOShape], ModelIOShape]:
        """
        Returns the output shape of the model

        :return: a list of shape tuple or shape tuple
        """
        if self._built:
            output_shapes = [ModelIOShape(tuple(o.shape[1:]), o.dtype) for o in self._get_keras_model().outputs]
        else:
            warning(
                "If the model and wrapper inputs mismatch it will only be noticed when building, "
                "before that output shape is an estimate and does not assert inputs."
            )
            output_shapes = [
                self.__compute_output_shape(o, self._wrapped_model, self._iterate_model_inputs(self._inputs))
                for o in self._wrapped_model.outputs
            ]

        return output_shapes if len(output_shapes) > 1 else output_shapes[0]

    def _get_keras_model(self) -> Model:
        """
        Get's the internal keras model that is being serialized

        :return: The internal keras model
        """
        assert self._keras_model

        return self._keras_model

    def _unitary_build(self, texts: SimpleTypeTexts) -> None:
        """
        Builds the model using its inputs

        :param texts: text list, ignored
        """
        if not self._built:
            inputs = []
            middle = []
            for model in self._iterate_model_inputs(self.inputs):
                input_shapes = (
                    [model.inputs_shape] if isinstance(model.inputs_shape, ModelIOShape) else model.inputs_shape
                )
                model_inps = [Input(shape.shape, dtype=shape.dtype) for shape in input_shapes]
                inputs += model_inps
                model_out = model(model_inps)
                if isinstance(model_out, list):
                    middle += model_out
                else:
                    middle += [model_out]
            if len(self._wrapped_model.inputs) == 1 and len(inputs) > 1:
                middle = Concatenate()(middle)
            outputs = self._call_keras_layer(self._wrapped_model, middle)
            self._keras_model = Model(inputs=inputs, outputs=outputs, name=repr(self))
            self._built = True

    def _find_text_inputs(self) -> List[TextRepresentation]:
        text_inputs = []

        # BFS until if finds text inputs
        model_queue: Queue = Queue()
        for model in self._iterate_model_inputs(self.inputs):
            if isinstance(model, TextRepresentation):
                text_inputs.append(model)
            model_queue.put_nowait(model)
        while not model_queue.empty():
            for model in self._iterate_model_inputs(model_queue.get_nowait().inputs):
                if isinstance(model, TextRepresentation):
                    text_inputs.append(model)
                model_queue.put_nowait(model)

        return text_inputs

    def dumps(self) -> bytes:
        """
        Dumps the model into bytes

        :return: a byte array
        """
        if self.has_multi_text_input():
            if self._built:
                inputs_bytes = []
                for name, inps in self.inputs:
                    name_inputs = []
                    for inp in inps:
                        if isinstance(inp, TextRepresentation):
                            name_inputs.append(inp.serialize())
                        else:
                            name_inputs += [
                                ti.serialize() for ti in inp._find_text_inputs()  # type: ignore[union-attr]
                            ]
                    inputs_bytes.append((name, name_inputs))
            else:
                inputs_bytes = [(name, [inp.serialize() for inp in inps]) for name, inps in self.inputs]
        else:
            model_inputs = cast(List[BaseModel], self.inputs)
            if self._keras_model:
                inputs_bytes = [inp.serialize() for inp in self._find_text_inputs()]
            else:
                inputs_bytes = [inp.serialize() for inp in model_inputs]
        wrapped_model_bytes = self.get_bytes_from_model(self._wrapped_model)
        keras_model_bytes = None
        if self._keras_model:
            keras_model_bytes = self.get_bytes_from_model(self._keras_model, copy=True)
        return pickle.dumps(
            (inputs_bytes, wrapped_model_bytes, keras_model_bytes, self._random_seed, self._built, self._frozen)
        )

    @classmethod
    def loads(cls, data: bytes) -> "KerasWrapper":
        """
        Loads a model

        :param data: the source bytes to load the model
        :return: a Serializable Model
        """
        inputs_bytes, wrapped_model_bytes, keras_model_bytes, random_seed, _built, _frozen = pickle.loads(data)
        if isinstance(inputs_bytes[0], tuple):
            inputs = [
                (name, [BaseModel.deserialize(inp_bytes) for inp_bytes in inps_bytes])
                for name, inps_bytes in inputs_bytes
            ]
        else:
            inputs = [BaseModel.deserialize(inp_bytes) for inp_bytes in inputs_bytes]
        obj = cls(inputs, cls.get_model_from_bytes(wrapped_model_bytes))
        if keras_model_bytes:
            obj._keras_model = cls.get_model_from_bytes(keras_model_bytes)
            obj._built = _built
        obj._random_seed = random_seed
        if _frozen:
            obj.freeze()
        return obj
