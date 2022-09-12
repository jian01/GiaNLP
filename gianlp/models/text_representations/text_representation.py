"""
Text input interface
"""
from abc import ABC, abstractmethod
from typing import List, Union, Callable, Optional
from multiprocessing import Pool
from functools import partial

from tensorflow import int32

from gianlp.models.base_model import BaseModel, ModelIOShape
from gianlp.types import ModelInputsWrapper, SimpleTypeTexts
from gianlp.config import get_default_jobs


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

    @staticmethod
    def parallel_tokenizer(
        text: str, tokenizer: Callable[[str], List[str]], sequence_maxlength: Optional[int] = None
    ) -> List[str]:
        """
        Parallelizable wrapper for the tokenizer

        :param text: the text to tokenize
        :param tokenizer: the tokenizer
        :param sequence_maxlength: optional sequence maxlength.
        :return: a list of lists with string tokens
        """
        if sequence_maxlength is None:
            return tokenizer(text)
        else:
            return tokenizer(text)[:sequence_maxlength]

    @staticmethod
    def tokenize_texts(
        texts: SimpleTypeTexts,
        tokenizer: Callable[[str], List[str]],
        sequence_maxlength: Optional[int] = None,
    ) -> List[List[str]]:
        """
        Function for tokenizing texts

        :param texts: the texts to tokenize
        :param tokenizer: the tokenizer
        :param sequence_maxlength: optional sequence maxlength.
        :return: a list of lists with string tokens
        """
        tokenizer = partial(
            TextRepresentation.parallel_tokenizer, tokenizer=tokenizer, sequence_maxlength=sequence_maxlength
        )
        if get_default_jobs() > 1:
            with Pool(get_default_jobs()) as p:
                return p.map(tokenizer, texts)
        return [tokenizer(text) for text in texts]
