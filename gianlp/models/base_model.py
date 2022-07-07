"""
Base model module
"""

import os
import pickle
import tarfile
from abc import abstractmethod, ABC
from io import BytesIO
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Union, List, Dict, Iterator, NamedTuple, Tuple, Optional

import numpy as np
import pandas as pd
# pylint: disable=no-name-in-module
from tensorflow import Tensor
from tensorflow.keras import backend as K
from tensorflow.keras.backend import clear_session, floatx
from tensorflow.keras.layers import Concatenate, TimeDistributed
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, load_model, clone_model

from gianlp.logging import warning
from gianlp.models._health_utils import get_dependencies_signature, warn_for_dependencies_signature
from gianlp.models._report_utils import model_list_to_summary_string

# pylint: enable=no-name-in-module

MODEL_DATA_PATHNAME = "model_data"

KerasInputOutput = Union[List[np.array], np.array]
SimpleTypeTexts = Union[List[str], pd.Series]
TextsInput = Union[SimpleTypeTexts, Dict[str, List[str]], pd.DataFrame]
ModelInputs = Union[List["BaseModel"], List[Tuple[str, List["BaseModel"]]]]


class ModelIOShape(NamedTuple):
    """
    Mopdel IO Shape object
    """

    shape: Tuple[int, ...]
    dtype: np.dtype = np.dtype(getattr(np, floatx()))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.shape}, {self.dtype.name}"


class BaseModel(ABC):
    """
    Serializable keras wrapper model class.

    Guarantees serializing and deserializing from and to objects with the same behaviour.
    All base models contains at least one model from Keras.

    :var _built: Indicates whether the model was built or not
    """

    _built: bool

    def __init__(self) -> None:
        self._built = False

    @property
    @abstractmethod
    def inputs(self) -> ModelInputs:
        """
        Method for getting all models that serve as input

        :return: a list or list of tuples containing BaseModel objects
        """

    def has_multi_text_input(self) -> bool:
        """
        Method for knowing if the model has multi input

        :return: True if the input of the model need multiple texts
        """
        input_models = self.inputs
        return input_models and (isinstance(input_models[0], tuple) or input_models[0].has_multi_text_input())

    @property
    @abstractmethod
    def outputs_shape(self) -> Union[List[ModelIOShape], ModelIOShape]:
        """
        Returns the output shape of the model

        :return: a list of shape tuple or shape tuple
        """

    @property
    @abstractmethod
    def inputs_shape(self) -> Union[List[ModelIOShape], ModelIOShape]:
        """
        Returns the shapes of the inputs of the model

        :return: a list of shape tuple or shape tuple
        """

    @property
    def weights_amount(self) -> Optional[int]:
        """
        Computes the total amount of weights

        :return: the total amount of weights or none if not built
        """
        if not self._built:
            return None
        return sum(K.count_params(w) for w in self._get_keras_model().weights)

    @property
    def trainable_weights_amount(self) -> Optional[int]:
        """
        Computes the total amount of trainable weights

        :return: the total amount of trainable weights or none if not built
        """
        if not self._built:
            return None
        return sum(K.count_params(w) for w in self._get_keras_model().trainable_weights)

    def __model_finder(self, model) -> List["BaseModel"]:
        """
        Finds all models chained to this one

        :param model: the initial model
        :return: a list of base models
        """
        models = [model]
        for m in self._iterate_model_inputs(model.inputs):
            models += self.__model_finder(m)
        return models

    def __str__(self) -> str:
        """
        Get's the model summary as string

        :return: the model summary
        """
        models = list(reversed(self.__model_finder(self)))
        seen_models = set()
        no_repetition_models = []
        for model in models:
            if repr(model) not in seen_models:
                no_repetition_models.append(model)
                seen_models.update([repr(model)])
        return model_list_to_summary_string(no_repetition_models)

    def __repr__(self) -> str:
        """
        Get's a string for identifying the model

        :return: an object identifier string
        """
        return f"{hex(id(self))[2:]}-{self.__class__.__name__}"

    @abstractmethod
    def _get_keras_model(self) -> Model:
        """
        Get's the internal keras model that is being serialized
        """

    @abstractmethod
    def _unitary_build(self, texts: SimpleTypeTexts) -> None:
        """
        Builds the model using its inputs

        :param texts: a text list for building if needed
        """

    @staticmethod
    def _iterate_model_inputs(model_inputs: ModelInputs) -> Iterator["BaseModel"]:
        """
        Iterates model inputs in the proper order
        :param model_inputs: the model inputs
        """
        if not model_inputs:
            return
        else:
            if isinstance(model_inputs[0], tuple):
                t: Tuple[str, List["BaseModel"]]
                for t in model_inputs:
                    for model in t[1]:
                        yield model
            else:
                for model in model_inputs:
                    yield model

    def build(self, texts: SimpleTypeTexts) -> None:
        """
        Builds the whole chain of models in a recursive manner using the functional API.
        Some operations may need the model to be built.

        :param texts: the texts for building if needed, some models have to learn from a sample corpus before working
        """
        if isinstance(texts, pd.Series):
            texts = texts.values.tolist()

        for model in self._iterate_model_inputs(self.inputs):
            model.build(texts)
        self._unitary_build(texts)

    @staticmethod
    def _call_keras_layer(keras_model: Model, inputs: Union[List[Layer], Layer]) -> Tensor:
        """
        Simple call for a keras layer for preprocessed inputs

        :param inputs: a Keras layer of list of Keras layers
        :return: the keras output
        """
        if isinstance(inputs, list):
            if len(inputs) > 1:
                return keras_model(inputs)
            inputs = inputs[0]

        assert len(keras_model.inputs) == 1

        if len(inputs.shape.as_list()) > len(keras_model.inputs[0].shape):
            for _ in range(len(inputs.shape.as_list()) - len(keras_model.inputs[0].shape)):
                keras_model = TimeDistributed(keras_model)
        return keras_model(inputs)

    def __call__(self, inputs: Union[List[Layer], Layer, np.array]) -> Union[Tensor, KerasInputOutput]:
        """
        Allows calling the internal Keras model supporting multiple formats

        :param inputs: either a keras layer, a list of keras layers or a numpy array
        :return: the output in keras format or numpy array if the input was a numpy array
        :raises:
            ValueError: If the model has not yet been built
        """
        if not self._built:
            raise ValueError("This model has not yet been built.")

        keras_model = self._get_keras_model()
        if type(inputs).__module__ == np.__name__ or (isinstance(inputs, list) and
                                                      type(inputs[0]).__module__ == np.__name__):
            return keras_model.predict(inputs)
        if isinstance(inputs, list) and len(keras_model.inputs) != len(inputs):
            outputs = []
            for layer in inputs:
                outputs.append(self._call_keras_layer(keras_model, layer))
            return Concatenate()(outputs)
        else:
            return self._call_keras_layer(keras_model, inputs)

    @abstractmethod
    def dumps(self) -> bytes:
        """
        Dumps the model into bytes

        :return: a byte array
        """

    def serialize(self) -> bytes:
        """
        Serializes the model to be deserialized with the deserialize method

        :return: a byte array
        """
        return pickle.dumps((self.__class__.__name__, get_dependencies_signature(), self.dumps()))

    @classmethod
    @abstractmethod
    def loads(cls, data: bytes) -> "BaseModel":
        """
        Loads a model

        :return: a Serializable Model
        """

    @classmethod
    def deserialize(cls, data: bytes) -> "BaseModel":
        """
        Deserializes a model

        :param data: the data for deserializing
        :return: a BaseModel object
        """
        serialized_items = pickle.loads(data)
        if len(serialized_items) == 2:
            name, data = serialized_items
        else:
            name, versions, data = serialized_items
            warn_for_dependencies_signature(versions)
        subclasses = {c.__name__: c for c in BaseModel.__subclasses__()}
        for sc in list(subclasses.values()):
            subclasses.update({c.__name__: c for c in sc.__subclasses__()})
        for sc in list(subclasses.values()):
            subclasses.update({c.__name__: c for c in sc.__subclasses__()})
        return subclasses[name].loads(data)

    @staticmethod
    def get_bytes_from_model(model: Model, copy: bool = False) -> bytes:
        """
        Transforms a keras model into bytes

        :param model: the keras model
        :param copy: whether to copy the model before saving.
        copying the model is needed for complex nested models because the keras save/load can fail
        :return: a byte array
        """
        model_path = TemporaryDirectory()

        if copy:
            model_copy = clone_model(model)
            """
            Since model_copy.set_weights(model.get_weights()) does not work as expected in some complex models:
                - We can't use save/load because when loading the model expects more weights for loading than the 
                ones saved.
                This is because some layers that are reused get saved as one variable but when loaded expects to have 
                two different variables.
                - We need to copy before saving, guaranteeing that the saved model respects the structure expected 
                when loaded. The copy emulates a save.
                - We can't copy using the simple get_weights/set_weights because the same bug that makes save/load 
                impossible happens
                
            Hence the next code block is needed for copying the weights:
            """
            if len(model.weights) == len(model_copy.weights):
                model_copy.set_weights(model.get_weights())
            else:  # This should never happen, but it does
                new_weights = []
                known_vars = {}
                shared_coerced = set()
                old_weigh_index = 0
                new_weigh_index = 0
                while len(new_weights) != len(model_copy.weights):
                    """
                    This 'if' may fail coercing the error in the particular case when the extra weight's shape missing
                    in the copy model matches the shape of the next weight. In that case there's no way this code
                    block won't raise an exception.
                    So, although this may fail in some border case, it is guaranteed that this will fail avoiding a
                    copy with wrong weights without noticing.
                    """
                    if model.weights[old_weigh_index].shape == model_copy.weights[new_weigh_index].shape:
                        new_weights.append(model.weights[old_weigh_index])
                        known_vars[model.weights[old_weigh_index].name] = model.weights[old_weigh_index]
                        old_weigh_index += 1
                        new_weigh_index += 1
                    else:
                        new_weights.append(known_vars[model_copy.weights[new_weigh_index].name])
                        shared_coerced.update([model_copy.weights[new_weigh_index].name])
                        new_weigh_index += 1
                for name in shared_coerced:
                    warning(
                        f"The weight {name} shared by multiple layers would be treated as non-shared when loaded, "
                        f"all the layers that shared that weight will fit independently from each other from now on."
                    )
                model_copy.set_weights([w.numpy() for w in new_weights])  # .weights is a read only attribute
            model = model_copy
            clear_session()

        model.save(model_path.name)

        with NamedTemporaryFile() as tar_temp_file:
            with tarfile.open(tar_temp_file.name, mode="w:gz") as archive:
                archive.add(model_path.name, arcname=MODEL_DATA_PATHNAME)

            model_path.cleanup()
            with open(tar_temp_file.name, "rb") as model_tarfile:
                model_bytes = model_tarfile.read()
        model_path.cleanup()
        return model_bytes

    @staticmethod
    def get_model_from_bytes(data: bytes) -> Model:
        """
        Given bytes from keras model serialized with get_bytes_from_model method
        returns the model

        :param data: the model bytes
        :return: a keras model
        """
        input_tarfile = tarfile.open(fileobj=BytesIO(data))
        with TemporaryDirectory() as output_dir:
            input_tarfile.extractall(output_dir)
            model = load_model(os.path.join(output_dir, MODEL_DATA_PATHNAME))
        return model

    @abstractmethod
    def preprocess_texts(self, texts: SimpleTypeTexts) -> KerasInputOutput:
        """
        Given texts returns the array representation needed for forwarding the
        keras model

        :param texts: the texts to preprocess
        :return: a numpy array or list of numpy arrays representing the texts
        """
