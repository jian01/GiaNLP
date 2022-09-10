"""
Text input interface
"""
from abc import ABC, abstractmethod
from typing import List, Union

from tensorflow import int32

from gianlp.models.base_model import BaseModel, ModelIOShape
from gianlp.types import ModelInputsWrapper


class TextRepresentation(BaseModel, ABC):
    """
    Text Representation class
    """

    @property
    def inputs(self) -> ModelInputsWrapper:
        """
        Method for getting all models that serve as input.
        All TextRepresentation have no models as an input.

        :return: a list or list of tuples containing BaseModel objects
        """

        return ModelInputsWrapper([])

    @property
    @abstractmethod
    def outputs_shape(self) -> ModelIOShape:
        """
        Returns the output shape of the model

        :return: a list of shape tuple or shape tuple
        """

    @property
    def inputs_shape(self) -> Union[List[ModelIOShape], ModelIOShape]:
        """
        Returns the shapes of the inputs of the model

        :return: a list of shape tuple or shape tuple
        """
        return ModelIOShape(self.outputs_shape.shape[:-1], int32)
