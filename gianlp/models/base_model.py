"""
Base model module
"""

import os
import pickle
import tarfile
from abc import abstractmethod, ABC
from io import BytesIO
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import NamedTuple, Optional, List, Tuple, Union

import numpy as np

# pylint: disable=no-name-in-module
from tensorflow import Tensor, RaggedTensor
from tensorflow.keras import backend as K
from tensorflow.keras.backend import clear_session, floatx
from tensorflow.keras.layers import Concatenate, TimeDistributed
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, load_model, clone_model

# pylint: enable=no-name-in-module


from gianlp.logging import warning
from gianlp.models._health_utils import get_dependencies_signature, warn_for_dependencies_signature
from gianlp.models._report_utils import compute_model_summary
from gianlp.types import KerasInputOutput, TextsInput, TextsInputWrapper, ModelInputsWrapper, SimpleTypeModels

MODEL_DATA_PATHNAME = "model_data"


class ModelIOShape(NamedTuple):
    """
    Mopdel IO Shape object
    """

    shape: Tuple[Union[int, None], ...]
    dtype: np.dtype = np.dtype(getattr(np, floatx()))
    ragged: bool = False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.shape}, {self.dtype.name}"


class BaseModel(ABC):
    """
    Serializable keras wrapper model class.

    Guarantees serializing and deserializing from and to objects with the same behaviour.
    All base models contains at least one model from Keras.
    """

    _built: bool

    def __init__(self) -> None:
        self._built = False

    @property
    @abstractmethod
    def inputs(self) -> ModelInputsWrapper:
        """
        Method for getting all models that serve as input

        :return: a ModelInputsWrapper
        """

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

    def __str__(self) -> str:
        """
        Gets the model summary as string

        :return: the model summary
        """
        return compute_model_summary(self)

    def __repr__(self) -> str:
        """
        Gets a string for identifying the model

        :return: an object identifier string
        """
        return f"{hex(id(self))[2:]}-{self.__class__.__name__}"

    @abstractmethod
    def _get_keras_model(self) -> Model:
        """
        Gets the internal keras model that is being serialized

        :return: The internal keras model
        """

    @abstractmethod
    def _unitary_build(self, texts: TextsInput) -> None:
        """
        Builds the model using its inputs

        :param texts: a text list for building if needed
        """

    def build(self, texts: TextsInput) -> None:
        """
        Builds the whole chain of models in a recursive manner using the functional API.
        Some operations may need the model to be built.

        :param texts: the texts for building if needed, some models have to learn from a sample corpus before working
        :raises ValueError: If the multi-text input keys do not match with the ones in a multi-text model
        """
        texts = TextsInputWrapper(texts)

        if self.inputs.is_multi_text():
            if texts.is_multi_text():
                if set(self.inputs.keys()) - set(texts.keys()):
                    raise ValueError(
                        f"The texts keys have missing multi-text model "
                        f"keys: {set(self.inputs.keys()) - set(texts.keys())}"
                    )
            for name, inps in self.inputs.items():
                if texts.is_multi_text():
                    current_texts = texts[name]
                else:
                    current_texts = texts.to_texts_inputs()
                for model in inps:
                    model.build(current_texts)
        else:
            for model in self.inputs:
                model.build(texts.to_texts_inputs())
        self._unitary_build(texts.flatten().to_texts_inputs())

    @staticmethod
    def _call_keras_layer(keras_model: Model, inputs: Union[List[Layer], Layer]) -> Tensor:
        """
        Simple call for a keras layer for preprocessed inputs

        :param keras_model: the keras Model to call
        :param inputs: a Keras layer or list of Keras layers
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

    def __call__(self, inputs: Union[List[Layer], Layer, np.ndarray]) -> KerasInputOutput:
        """
        Allows calling the internal Keras model supporting multiple formats

        :param inputs: either a keras layer, a list of keras layers or a numpy array
        :return: the output in keras format or numpy array if the input was a numpy array
        :raises ValueError: If the model has not yet been built
        """
        if not self._built:
            raise ValueError("This model has not yet been built.")

        keras_model = self._get_keras_model()
        if (
            type(inputs).__module__ == np.__name__
            or (isinstance(inputs, list) and type(inputs[0]).__module__ == np.__name__)
            or isinstance(inputs, RaggedTensor)
        ):
            return keras_model.predict(inputs)
        if isinstance(inputs, list) and len(keras_model.inputs) != len(inputs):
            outputs = []
            for layer in inputs:
                outputs.append(self._call_keras_layer(keras_model, layer))
            return Concatenate()(outputs)
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

        :param data: the source bytes to load the model
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
        # support for legacy serialized models
        if len(serialized_items) == 2:  # pragma: no cover
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
                new_weights: List = []
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
    def preprocess_texts(self, texts: TextsInput) -> KerasInputOutput:
        """
        Given texts returns the array representation needed for forwarding the
        keras model

        :param texts: the texts to preprocess
        :return: a numpy array or list of numpy arrays representing the texts
        """
