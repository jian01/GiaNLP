"""
Trainable model Interface
"""
import types
from abc import ABC
from typing import Union, Tuple, Optional, Any, Callable, List, overload, cast

import numpy as np

# pylint: disable=no-name-in-module
from tensorflow.keras.callbacks import Callback, History
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.utils import OrderedEnqueuer, GeneratorEnqueuer
from tensorflow.keras.utils import Sequence as KerasSequence

# pylint: enable=no-name-in-module
from tqdm import tqdm

from gianlp.logging import warning
from gianlp.models.base_model import BaseModel, SimpleTypeModels
from gianlp.types import (
    TextsInput,
    KerasInputOutput,
    YielderGenerator,
    ModelFitTuple,
    KerasModelFitTuple,
    TextsInputWrapper,
    ModelOutputsWrapper,
)
from gianlp.utils import Sequence


class SequenceWrapper(KerasSequence, ABC):
    """
    Sequence wrapper interface
    """

    def __init__(self, sequence: Sequence, preprocessor: Callable[[TextsInput], KerasInputOutput]):
        self.sequence = sequence
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.sequence)


class TrainSequenceWrapper(SequenceWrapper):
    """
    Keras sequence generator for training wrapping utils.Sequence
    """

    def __getitem__(self, index) -> ModelFitTuple:
        x, y = self.sequence.__getitem__(index)
        x = self.preprocessor(x)
        return x, y

    def __iter__(self) -> YielderGenerator[ModelFitTuple]:
        """
        # noqa: DAR202

        Create a generator that iterate over the Sequence.
        :return: The generator
        """
        for i in range(len(self)):
            yield self.__getitem__(i)

    def on_epoch_end(self) -> None:
        """
        Method called at the end of every epoch.
        """
        self.sequence.on_epoch_end()


class PredictSequenceWrapper(SequenceWrapper):
    """
    Keras sequence generator for predicting wrapping utils.Sequence
    """

    def __getitem__(self, index) -> TextsInput:
        x = self.sequence.__getitem__(index)
        if isinstance(x, tuple):
            x = x[0]
        x = self.preprocessor(x)
        return x

    def __iter__(self) -> YielderGenerator[TextsInput]:
        """
        # noqa: DAR202

        Create a generator that iterate over the Sequence.
        :return: The generator
        """
        for i in range(len(self)):
            x = self.__getitem__(i)
            if isinstance(x, tuple):
                x = x[0]
            yield x


class TrainableModel(BaseModel, ABC):
    """
    Class for models that are trainable.

    It mimics Keras API.

    :var _random_seed: random_seed used in training and can be used for any random process of subclasses
    :var _frozen: if the model was frozen, this is needed for older tensorflow versions
    """

    _random_seed: int
    _frozen: int

    def __init__(self, random_seed: int = 42):
        super().__init__()
        self._random_seed = random_seed
        self._frozen = False

    def preprocess_texts(self, texts: TextsInput) -> KerasInputOutput:
        """
        Given texts returns the array representation needed for forwarding the
        keras model

        :param texts: the texts to preprocess
        :return: a numpy array or list of numpy arrays representing the texts
        :raises ValueError:
            - When the model is multi-text and x is not a dict or dataframe
            - When the model is not multi-text and x is a dict or dataframe
        """
        texts = TextsInputWrapper(texts)

        if self.has_multi_text_input():
            if not texts.is_multi_text():
                raise ValueError("The model has multi-text input but there's only one type of text to preprocess.")
        else:
            if texts.is_multi_text():
                raise ValueError("The model has input of only one type of text but multiple texts where feeded.")
        texts_preprocessed: List[np.ndarray] = []
        input_models = self.inputs
        if isinstance(input_models[0], tuple):
            for name, inps in input_models:
                for inp in inps:
                    result = inp.preprocess_texts(texts[name])
                    if isinstance(result, list):
                        texts_preprocessed += result
                    else:
                        texts_preprocessed.append(result)
        else:
            input_models = cast(SimpleTypeModels, input_models)
            for inp in input_models:
                result = inp.preprocess_texts(texts.to_texts_inputs())
                if isinstance(result, list):
                    texts_preprocessed += result
                else:
                    texts_preprocessed.append(result)
        if len(texts_preprocessed) == 1:
            return texts_preprocessed[0]
        return texts_preprocessed

    def compile(
        self,
        optimizer: Union[str, Optimizer] = "rmsprop",
        loss: Optional[Union[str, Loss]] = None,
        metrics: Optional[List[Union[str, Metric]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Compiles the Keras model and prepares the text inputs to be used

        :param optimizer: optimizer for training
        :param loss: loss for training
        :param metrics: metrics to use while training
        :param **kwargs: accepts any other parameters for use in Keras Model.compile API
        :raises AssertionError:
                - When the model is not built
        """
        assert self._built
        self._get_keras_model().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    @staticmethod
    def __shuffle_fit_data(
        x: TextsInputWrapper, y: ModelOutputsWrapper
    ) -> Tuple[TextsInputWrapper, ModelOutputsWrapper]:
        """
        Shuffles the fit data
        :param x: x fit data
        :param y: y fit data
        :return: shuffled fit data
        """
        perm = np.random.permutation(len(x))
        x = x[perm]
        y = y[perm]
        return x, y

    @overload
    def _fit_generator(
        self,
        data: YielderGenerator[ModelFitTuple],
        batch_size: int = 32,
    ) -> YielderGenerator[ModelFitTuple]:
        ...

    @overload
    def _fit_generator(
        self,
        data: Union[TextsInputWrapper, ModelOutputsWrapper],
        batch_size: int = 32,
    ) -> YielderGenerator[KerasModelFitTuple]:
        ...

    def _fit_generator(
        self,
        data: Union[YielderGenerator[ModelFitTuple], Tuple[TextsInputWrapper, ModelOutputsWrapper]],
        batch_size: int,
    ) -> YielderGenerator[KerasModelFitTuple]:
        """
        Internal generator for training

        :param data: generator of tuples (x,y) or tuple (x,y) with the training data
        :param batch_size: batch size for feeding the training. Ignored if data is a generator.
        """
        iter_range = None
        while True:
            if isinstance(data, types.GeneratorType):
                batch_x, batch_y = next(data)
                inputs = self.preprocess_texts(batch_x)
            else:
                data = cast(Tuple[TextsInputWrapper, ModelOutputsWrapper], data)
                if not iter_range:
                    x, y = data
                    x, y = self.__shuffle_fit_data(x, y)
                    iter_range = iter(range(0, len(x), batch_size))
                try:
                    i = next(iter_range)
                # pytest fails to record some coverage
                except StopIteration:  # pragma: no cover
                    iter_range = iter(range(0, len(x), batch_size))
                    i = next(iter_range)
                    x, y = self.__shuffle_fit_data(x, y)
                batch_x, batch_y = x[i : i + batch_size], y[i : i + batch_size]
                if len(batch_x) < batch_size:  # pragma: no cover
                    sliced_extra = x[0 : batch_size - len(batch_x)]
                    batch_x += sliced_extra
                    batch_y += y[0 : batch_size - len(batch_y)]
                inputs = self.preprocess_texts(batch_x.to_texts_inputs())
                batch_y = batch_y.to_model_outputs()

            yield inputs, batch_y

    @overload
    def _build_fit_generator(self, data: Tuple[Union[YielderGenerator[ModelFitTuple], Sequence], None], batch_size=...):
        ...

    @overload
    def _build_fit_generator(self, data: Tuple[TextsInputWrapper, ModelOutputsWrapper], batch_size=...):
        ...

    def _build_fit_generator(
        self,
        data: Tuple[Union[YielderGenerator[ModelFitTuple], TextsInputWrapper, Sequence], Optional[ModelOutputsWrapper]],
        batch_size: int,
        steps_per_epoch: Optional[int],
        max_queue_size: int,
        workers: int,
        use_multiprocessing: bool,
    ) -> Tuple[Union[YielderGenerator[KerasModelFitTuple], TrainSequenceWrapper], Optional[int]]:
        """

        :param data: tuple of (x,y), where:
            * x could be a generator and y is None
            * x could be a Sequence and y is None
            * x could be text input and y an array
        :param batch_size: batch size for feeding the training. Ignored if data is a generator.
        :param steps_per_epoch: the steps per epoch for the generator
        :param max_queue_size: Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
        :param workers: Maximum number of processes to spin up when using process-based threading. If unspecified,
        workers will default to 1.
        :param use_multiprocessing: If True, use process-based threading. If unspecified, use_multiprocessing will
        default to False. Note that because this implementation relies on multiprocessing, you should not pass
        non-picklable arguments to the generator as they can't be passed easily to children processes.
        :return: a generator for using in keras fit method and the steps per epoch to use
        """
        generator: Union[YielderGenerator[KerasModelFitTuple], TrainSequenceWrapper]

        x, y = data
        if isinstance(x, Sequence):
            generator = TrainSequenceWrapper(x, self.preprocess_texts)
            if use_multiprocessing:
                enq = OrderedEnqueuer(generator, use_multiprocessing=True)
                enq.start(workers=workers, max_queue_size=max_queue_size)
                generator = cast(YielderGenerator[KerasModelFitTuple], enq.get())
            return generator, steps_per_epoch
        elif isinstance(x, types.GeneratorType):
            generator = self._fit_generator(x, batch_size)
            if use_multiprocessing:
                enq = GeneratorEnqueuer(generator, use_multiprocessing=True, random_seed=self._random_seed)
                enq.start(workers=workers, max_queue_size=max_queue_size)
                generator = cast(YielderGenerator[KerasModelFitTuple], enq.get())
            return generator, steps_per_epoch
        if use_multiprocessing:
            warning(
                "Can't use multiprocessing with already generated data. "
                "This parameter only affects generators or utils.Sequence objects."
            )
        x = cast(TextsInputWrapper, x)
        y = cast(ModelOutputsWrapper, y)
        generator = self._fit_generator((x, y), batch_size)
        steps_per_epoch = len(x) // batch_size
        return generator, steps_per_epoch

    @overload
    def fit(self, x: TextsInput, y: KerasInputOutput = ...):
        ...

    @overload
    def fit(self, x: Union[YielderGenerator[ModelFitTuple], Sequence], y: None = ...):
        ...

    def fit(
        self,
        x: Union[YielderGenerator[ModelFitTuple], TextsInput, Sequence],
        y: Optional[KerasInputOutput] = None,
        batch_size: int = 32,
        epochs: int = 1,
        verbose: Union[str, int] = "auto",
        callbacks: List[Callback] = None,
        validation_split: float = 0.0,
        validation_data: Optional[Union[YielderGenerator[ModelFitTuple], ModelFitTuple, Sequence]] = None,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
        max_queue_size: int = 10,
        workers: int = 1,
        use_multiprocessing: bool = False,
        **kwargs,
    ) -> History:
        """
        Fits the model

        :param x: Input data. Could be:

            * A generator that yields (x, y) where x is any valid format for x and y is the target numpy array
            * A :class:`gianlp.utils.Sequence` object that generates (x, y) where x is any valid format for x and y is\
            the target output
            * A list of texts
            * A pandas Series
            * A pandas Dataframe
            * A dict of lists containing texts

        :param y: Target, ignored if x is a generator. Numpy array.
        :param batch_size: Batch size for training, ignored if x is a generator or a :class:`gianlp.utils.Sequence`
        :param epochs: Amount of epochs to train
        :param verbose: verbose mode for Keras training
        :param callbacks: list of Callback objects for Keras model
        :param validation_split: the proportion of data to use for validation, ignored if x is a generator.
            Takes the last elements of x and y. Ignored if x is a generator or a :class:`gianlp.utils.Sequence` object
        :param validation_data: Validation data. Could be:

            *. A tuple containing (x, y) where x is a any valid format for x and y is the target numpy array
            *. A generator that yields (x, y) where x is a any valid format for x and y is the target numpy array
            *. :class:`gianlp.utils.Sequence` object that generates (x, y) where x is any valid format for x and y is\
            the target output

        :param steps_per_epoch: Amount of generator steps to consider an epoch as finished. Ignored if x is not a
            generator
        :param validation_steps: Amount of generator steps to consider to feed each validation evaluation.
            Ignored if validation_data is not a generator
        :param max_queue_size: Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
        :param workers: Maximum number of processes to spin up when using process-based threading. If unspecified,
            workers will default to 1.
        :param use_multiprocessing: If True, use process-based threading. If unspecified, use_multiprocessing will
            default to False. Note that because this implementation relies on multiprocessing, you should not pass
            non-picklable arguments to the generator as they can't be passed easily to children processes.
        :param **kwargs: extra arguments to give to keras.models.Model.fit
        :return: A History object. Its History.history attribute is a record of training loss values and metrics values
            at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """
        train_generator: Union[TrainSequenceWrapper, YielderGenerator[KerasModelFitTuple]]

        np.random.seed(self._random_seed)

        if not validation_data is None:
            if not isinstance(validation_data, Sequence) and not isinstance(validation_data, types.GeneratorType):
                validation_data = cast(ModelFitTuple, validation_data)
                validation_data = TextsInputWrapper(validation_data[0]), ModelOutputsWrapper(validation_data[1])
            else:
                validation_data = (validation_data, None)

        if not isinstance(x, Sequence) and not isinstance(x, types.GeneratorType):
            x = TextsInputWrapper(x)
            y = cast(KerasInputOutput, y)
            y = ModelOutputsWrapper(y)
            if validation_split > 0 and validation_data is None:
                valid_amount = int(round(validation_split * len(x)))
                validation_data = (
                    x[-valid_amount:],
                    y[-valid_amount:],
                )
                x = x[:-valid_amount]
                y = y[:-valid_amount]
        train_generator, steps_per_epoch = self._build_fit_generator(
            (x, y), batch_size, steps_per_epoch, max_queue_size, workers, use_multiprocessing
        )
        valid_generator = None
        if not validation_data is None:
            valid_generator, validation_steps = self._build_fit_generator(
                validation_data, batch_size, validation_steps, max_queue_size, workers, use_multiprocessing
            )

        trainable_model = self._get_keras_model()
        return trainable_model.fit(
            train_generator,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=valid_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            max_queue_size=max_queue_size,
            workers=(1 if use_multiprocessing else workers),
            use_multiprocessing=False,
            **kwargs,
        )

    def _predict_generator(
        self, x: Union[YielderGenerator[TextsInput], TextsInputWrapper], inference_batch: int
    ) -> YielderGenerator[KerasInputOutput]:
        """
        # noqa: DAR202

        Internal generator for predictions, transforms text input into numerical input

        :param x: generator of x or x text inputs
        :param inference_batch: inference batch size, ignored if x is generator
        :return: a generator of numerical input of the model
        """
        if isinstance(x, types.GeneratorType):
            while True:
                try:
                    texts = next(x)
                except StopIteration:
                    break
                if isinstance(texts, tuple):
                    # ignore a generator that also feeds labels
                    texts = texts[0]
                yield self.preprocess_texts(texts)
        else:
            x = cast(TextsInputWrapper, x)
            for i in range(0, len(x), inference_batch):
                batch_x = x[i : i + inference_batch]
                if len(batch_x) < inference_batch:
                    sliced_extra = x[0 : inference_batch - len(batch_x)]
                    batch_x += sliced_extra
                yield self.preprocess_texts(batch_x.to_texts_inputs())

    @overload
    def predict(
        self, x: Union[YielderGenerator[TextsInput], TextsInput], inference_batch: int = ...
    ) -> KerasInputOutput:
        ...

    @overload
    def predict(self, x: Sequence, inference_batch: int = ...) -> KerasInputOutput:
        ...

    def predict(
        self,
        x: Union[YielderGenerator[TextsInput], TextsInput, Sequence],
        inference_batch: int = 256,
        steps: Optional[int] = None,
        max_queue_size: int = 10,
        workers: int = 1,
        use_multiprocessing: bool = False,
        verbose: int = 0,
    ) -> KerasInputOutput:
        """
        Predicts using the model

        :param x: Could be:

            * A list of texts
            * A pandas Series
            * A pandas Dataframe
            * A dict of lists containing texts
            * A generator of any of the above formats
            * A :class:`gianlp.utils.Sequence` object that generates batches of text

        :param inference_batch: the prediction is made in batches for saving ram, this is the batch size used.
            ignored if x is a generator or a :class:`gianlp.utils.Sequence`
        :param steps: steps for the generator, ignored if x is not a generator
        :param max_queue_size: Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
        :param workers: Maximum number of processes to spin up when using process-based threading. If unspecified,
            workers will default to 1.
        :param use_multiprocessing: If True, use process-based threading. If unspecified, use_multiprocessing will
            default to False. Note that because this implementation relies on multiprocessing, you should not pass
            non-picklable arguments to the generator as they can't be passed easily to children processes.
        :param verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = single line.
        :return: the output of the keras model
        :raises ValueError: If a generator is given as x but no step amount is specified
        """
        predict_generator: Union[PredictSequenceWrapper, YielderGenerator[KerasInputOutput]]

        preds = None
        if isinstance(x, Sequence):
            steps = len(x) if not steps else min(steps, len(x))
            predict_generator = PredictSequenceWrapper(x, self.preprocess_texts)
            if use_multiprocessing:
                enq = OrderedEnqueuer(predict_generator, use_multiprocessing=True)
                enq.start(workers=workers, max_queue_size=max_queue_size)
                predict_generator = enq.get()
            else:
                predict_generator = iter(predict_generator)
        else:
            if not isinstance(x, types.GeneratorType):
                x = TextsInputWrapper(x)
            predict_generator = self._predict_generator(x, inference_batch)
            if use_multiprocessing and isinstance(x, types.GeneratorType):
                warning(
                    "Keras API allows prediction generators with multiprocessing, so does this method, "
                    "but be aware this completely looses track of which predictions are from which label "
                    "since order will be lost by concurrency. We recommend using a utils.Sequence object for"
                    " multiprocessing."
                )
                enq = GeneratorEnqueuer(predict_generator, use_multiprocessing=True, random_seed=self._random_seed)
                enq.start(workers=workers, max_queue_size=max_queue_size)
                predict_generator = enq.get()
            if not steps:
                if isinstance(x, types.GeneratorType):
                    raise ValueError("For using a generator the steps to use need to be specified.")
                steps = len(x) // inference_batch + (1 if len(x) % inference_batch != 0 else 0)
        for i in tqdm(range(steps), total=steps, disable=(True if verbose != 2 else False)):
            try:
                batch = next(predict_generator)
            except StopIteration:
                warning("Generator stopped before reaching the steps specified.")
                break
            pred_batch = self._get_keras_model().predict_on_batch(batch)
            if not preds:
                preds = ModelOutputsWrapper(pred_batch)
            else:
                preds += ModelOutputsWrapper(pred_batch)
            if verbose == 1:
                print(f"{i}/{steps} Batch predicted")
        if not isinstance(x, types.GeneratorType) and not isinstance(x, Sequence):
            preds = cast(ModelOutputsWrapper, preds)
            return preds[: len(x)].to_model_outputs()
        return preds.to_model_outputs()  # type: ignore[union-attr]

    def freeze(self) -> None:
        """
        Freezes the model weights

        :raises ValueError: When the model is not built
        """
        if not self._built:
            raise ValueError("Can't freeze a model that has not been built")
        model = self._get_keras_model()
        for k, _ in model._get_trainable_state().items():
            k.trainable = False
        for inp in self._iterate_model_inputs(self.inputs):
            if isinstance(inp, TrainableModel):
                inp.freeze()
        self._frozen = True
