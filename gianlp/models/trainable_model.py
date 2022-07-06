"""
Trainable model Interface
"""
import types
from abc import ABC
from typing import Union, Generator, List, Tuple, Optional, Any, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.callbacks import Callback, History
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.utils import Sequence as KerasSequence
from tensorflow.keras.utils import OrderedEnqueuer, GeneratorEnqueuer

from gianlp.logging import warning
from gianlp.models.base_model import BaseModel, KerasInputOutput, TextsInput
from gianlp.utils import Sequence


class TrainSequenceWrapper(KerasSequence):
    """
    Keras sequence generator for training wrapping utils.Sequence
    """

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index) -> Tuple[TextsInput, KerasInputOutput]:
        x, y = self.sequence.__getitem__(index)
        x = self.preprocessor(x)
        return x, y

    def __init__(self, sequence: Sequence,
                 preprocessor: Callable[[TextsInput], KerasInputOutput]):
        self.sequence = sequence
        self.preprocessor = preprocessor

    def __iter__(self) -> Generator[Tuple[TextsInput, KerasInputOutput], None, None]:
        """
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


class PredictSequenceWrapper(TrainSequenceWrapper):
    """
    Keras sequence generator for predicting wrapping utils.Sequence
    """

    def __getitem__(self, index) -> TextsInput:
        x = self.sequence.__getitem__(index)
        if isinstance(x, tuple):
            x = x[0]
        x = self.preprocessor(x)
        return x

    def __iter__(self) -> Generator[TextsInput, None, None]:
        """
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
    """

    _random_seed: int

    def __init__(self, random_seed: int = 42):
        super().__init__()
        self._random_seed = random_seed

    def preprocess_texts(self, texts: TextsInput) -> KerasInputOutput:
        """
        Given texts returns the array representation needed for forwarding the
        keras model

        :param texts: the texts to preprocess
        :return: a numpy array or list of numpy arrays representing the texts
        :raises:
            ValueError:
            - When the model is multi-text and x is not a dict or dataframe
            - When the model is not multi-text and x is a dict or dataframe
        """
        if isinstance(texts, pd.Series):
            texts = texts.values.tolist()
        if isinstance(texts, pd.DataFrame):
            texts = texts.to_dict("list")

        if self.has_multi_text_input():
            if isinstance(texts, list):
                raise ValueError("The model has multi-text input but there's only one type of text to preprocess.")
        else:
            if isinstance(texts, dict):
                raise ValueError("The model has input of only one type of text but multiple texts where feeded.")
        texts_preprocessed = []
        input_models = self.inputs
        if isinstance(input_models[0], tuple):
            for name, inps in self.inputs:
                for inp in inps:
                    result = inp.preprocess_texts(texts[name])
                    if result is not None:
                        texts_preprocessed.append(result)
        else:
            for inp in self.inputs:
                result = inp.preprocess_texts(texts)
                if result is not None:
                    texts_preprocessed.append(result)
        if len(texts_preprocessed) == 1:
            return texts_preprocessed[0]
        return texts_preprocessed

    def compile(
            self,
            optimizer: Union[str, Optimizer] = "rmsprop",
            loss: Optional[Union[str, Loss]] = None,
            metrics: Optional[List[Union[str, Metric]]] = None,
            **kwargs: Any
    ) -> None:
        """
        Compiles the Keras model and prepares the text inputs to be used

        :param optimizer: optimizer for training
        :param loss: loss for training
        :param metrics: metrics to use while training
        :param kwargs: accepts any other parameters for use in Keras Model.compile API
        :raises:
            AssertionError:
                - When the model is not built
        """
        assert self._built
        self._get_keras_model().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    @staticmethod
    def __slice_texts_input(text_input: TextsInput, low: int, high: int) -> TextsInput:
        """
        Slices texts inputs

        :param text_input: the texts input
        :param low: the low index
        :param high: the high index
        :return: the sliced texts input
        """
        if isinstance(text_input, pd.DataFrame):
            text_input = text_input.to_dict("list")
        if isinstance(text_input, pd.Series):
            text_input = text_input.values.tolist()

        if isinstance(text_input, dict):
            return {k: v[low:high] for k, v in text_input.items()}
        else:
            return text_input[low:high]

    @staticmethod
    def __texts_input_length(text_input: TextsInput) -> int:
        """
        Computes the texts inputs length
        :param text_input: the texts input
        :return: the length
        """
        if isinstance(text_input, pd.DataFrame):
            text_input = text_input.to_dict("list")
        if isinstance(text_input, pd.Series):
            text_input = text_input.values.tolist()

        if isinstance(text_input, dict):
            return len(text_input[list(text_input.keys())[0]])
        else:
            return len(text_input)

    @staticmethod
    def __shuffle_fit_data(data: Tuple[TextsInput, KerasInputOutput]) -> Tuple[TextsInput, KerasInputOutput]:
        """
        Shuffles the fit data
        :param data: fit data
        :return: shuffled fit data
        """
        x, y = data
        if isinstance(x, pd.DataFrame):
            x = x.to_dict("list")
        if isinstance(x, dict):
            perm = np.random.permutation(len(y))
            x = {k: np.asarray(v)[perm].tolist() for k, v in x.items()}
            y = y[perm]
        else:
            perm = np.random.permutation(len(x))
            x = np.asarray(x)
            x, y = x[perm].tolist(), y[perm]
        return x, y

    def _fit_generator(
            self,
            data: Union[
                Generator[Tuple[TextsInput, KerasInputOutput], None, None], Tuple[TextsInput, KerasInputOutput]],
            batch_size: int = 32,
    ) -> Generator[Tuple[KerasInputOutput, KerasInputOutput], None, None]:
        """
        Internal generator for training

        :param data: generator of tuples (x,y) or tuple (x,y) with the training data
        :param batch_size: batch size for feeding the training. Ignored if data is a generator.
        """
        iter_range = None
        while True:
            if isinstance(data, types.GeneratorType):
                batch_x, batch_y = next(data)
            else:
                if not iter_range:
                    data = self.__shuffle_fit_data(data)
                    iter_range = iter(range(0, self.__texts_input_length(data[0]), batch_size))
                try:
                    i = next(iter_range)
                # pytest fails to record some coverage
                except StopIteration:  # pragma: no cover
                    iter_range = iter(range(0, self.__texts_input_length(data[0]), batch_size))
                    i = next(iter_range)
                    data = self.__shuffle_fit_data(data)
                batch_x, batch_y = self.__slice_texts_input(data[0], i, i + batch_size), data[1][i: i + batch_size]
                if self.__texts_input_length(batch_x) < batch_size:  # pragma: no cover
                    sliced_extra = self.__slice_texts_input(data[0], 0, batch_size - self.__texts_input_length(batch_x))
                    if isinstance(batch_x, dict):
                        batch_x = {k: v + sliced_extra[k] for k, v in batch_x.items()}
                    else:
                        batch_x += sliced_extra
                    batch_y = batch_y.tolist()
                    batch_y += data[1][0: batch_size - len(batch_y)].tolist()
                    batch_y = np.asarray(batch_y)

            inputs = self.preprocess_texts(batch_x)

            yield inputs, batch_y

    def _get_validation_generator(self, validation_data: Optional[
        Union[Generator[Tuple[TextsInput, KerasInputOutput], None, None], Tuple[
            TextsInput, KerasInputOutput]]], batch_size, validation_steps):
        if validation_data is None:
            valid_generator = None
        elif isinstance(validation_data, types.GeneratorType):
            valid_generator = self._fit_generator(validation_data)
        else:
            valid_generator = self._fit_generator(validation_data, batch_size)
            validation_steps = self.__texts_input_length(validation_data[0]) // batch_size
        return valid_generator, validation_steps

    def fit(
            self,
            x: Union[Generator[Tuple[TextsInput, KerasInputOutput], None, None],
                     TextsInput,
                     Sequence] = None,
            y: Optional[np.array] = None,
            batch_size: int = 32,
            epochs: int = 1,
            verbose: Union[str, int] = "auto",
            callbacks: List[Callback] = None,
            validation_split: Optional[float] = 0.0,
            validation_data: Optional[
                Union[Generator[Tuple[TextsInput, KerasInputOutput], None, None], Tuple[TextsInput, KerasInputOutput]]
            ] = None,
            steps_per_epoch: Optional[int] = None,
            validation_steps: Optional[int] = None,
            max_queue_size: int = 10,
            workers: int = 1,
            use_multiprocessing: bool = False
    ) -> History:
        """
        Fits the model

        :param x: Input data. Could be:
            1. A generator that yields (x, y) where x is any valid format for x and y is the target numpy array
            2. A gianlp.utils.Sequence object that generates (x, y) where x is any valid format for x and y is the target numpy array
            3. A list of texts
            4. A pandas Series
            5. A pandas Dataframe
            6. A dict of lists containing texts
        :param y: Target, ignored if x is a generator. Numpy array.
        :param batch_size: Batch size for training, ignored if x is a generator or a gianlp.utils.Sequence
        :param epochs: Amount of epochs to train
        :param verbose: verbose mode for Keras training
        :param callbacks: list of Callback objects for Keras model
        :param validation_split: the proportion of data to use for validation, ignored if x is a generator
        :param validation_data: Validation data. Could be:
            1. A tuple containing (x, y) where x is a any valid format for x and y is the target numpy array
            2. A generator that yields (x, y) where x is a any valid format for x and y is the target numpy array
        :param steps_per_epoch: Amount of generator steps to consider an epoch as finished. Ignored if x is not a
        generator
        :param validation_steps: Amount of generator steps to consider to feed each validation evaluation.
                                Ignored if validation_data is not a generator
        :param max_queue_size: Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
        :param workers: Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
        :param use_multiprocessing: If True, use process-based threading. If unspecified, use_multiprocessing will default to False. Note that because this implementation relies on multiprocessing, you should not pass non-picklable arguments to the generator as they can't be passed easily to children processes.
        :return: A History object. Its History.history attribute is a record of training loss values and metrics values
        at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """
        np.random.seed(self._random_seed)
        if isinstance(x, Sequence):
            train_generator = TrainSequenceWrapper(x, self.preprocess_texts)
            if use_multiprocessing:
                enq = OrderedEnqueuer(train_generator, use_multiprocessing=True)
                enq.start(workers=workers, max_queue_size=max_queue_size)
                train_generator = enq.get()
        elif isinstance(x, types.GeneratorType):
            train_generator = self._fit_generator(x)
            if use_multiprocessing:
                enq = GeneratorEnqueuer(train_generator, use_multiprocessing=True,
                                        random_seed=self._random_seed)
                enq.start(workers=workers, max_queue_size=max_queue_size)
                train_generator = enq.get()
        else:
            if use_multiprocessing:
                raise ValueError("Can't use multiprocessing with already generatred data.")
            train_data = (x, y)
            if validation_split > 0 and not validation_data:
                x, y = self.__shuffle_fit_data((x, y))

                valid_amount = int(round(validation_split * self.__texts_input_length(x)))
                validation_data = (
                    self.__slice_texts_input(x, -valid_amount, self.__texts_input_length(x)), y[-valid_amount:])
                train_data = (self.__slice_texts_input(x, 0, -valid_amount), y[:-valid_amount])
            train_generator = self._fit_generator(train_data, batch_size)
            steps_per_epoch = self.__texts_input_length(train_data[0]) // batch_size

        valid_generator, validation_steps = self._get_validation_generator(validation_data, batch_size, validation_steps)

        if use_multiprocessing and isinstance(validation_data, types.GeneratorType):
            enq = GeneratorEnqueuer(valid_generator, use_multiprocessing=True,
                                    random_seed=self._random_seed)
            enq.start(workers=workers, max_queue_size=max_queue_size)
            valid_generator = enq.get()

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
            use_multiprocessing=False
        )

    def _predict_generator(self, x: Union[Generator[TextsInput, None, None],
                                          TextsInput],
                           inference_batch: int) -> Generator[KerasInputOutput, None, None]:
        """
        Internal generator for predictions

        :param x: generator of x or x text inputs
        :param inference_batch: inference batch size, ignored if x is generator
        :return:
        """
        if isinstance(x, types.GeneratorType):
            while True:
                try:
                    texts = next(x)
                except StopIteration as e:
                    break
                if isinstance(texts, tuple):
                    # ignore a generator that also feeds labels
                    texts = texts[0]
                yield self.preprocess_texts(texts)
        else:
            for i in range(0, self.__texts_input_length(x), inference_batch):
                batch_x = self.__slice_texts_input(x, i, i + inference_batch)
                if self.__texts_input_length(batch_x) < inference_batch:
                    sliced_extra = self.__slice_texts_input(x, 0, inference_batch - self.__texts_input_length(batch_x))
                    if isinstance(batch_x, dict):
                        batch_x = {k: v + sliced_extra[k] for k, v in batch_x.items()}
                    else:
                        batch_x += sliced_extra
                yield self.preprocess_texts(batch_x)

    @staticmethod
    def __merge_preds(old_preds: KerasInputOutput, new_preds: KerasInputOutput) -> KerasInputOutput:
        """
        Merges an old list of predictions with a new one
        :param old_preds: old predictions
        :param new_preds: new predictions
        :return: merged predictions
        """
        if old_preds is None:
            return new_preds
        if isinstance(new_preds, list):
            for i in range(len(old_preds)):
                old_preds[i] = np.concatenate([old_preds[i], new_preds[i]])
        else:
            old_preds = np.concatenate([old_preds, new_preds])
        return old_preds

    def predict(
            self,
            x: Union[Generator[TextsInput, None, None],
                     TextsInput, Sequence],
            inference_batch: int = 256,
            steps: Optional[int] = None,
            max_queue_size: int = 10,
            workers: int = 1,
            use_multiprocessing: bool = False,
            verbose: int = 0
    ) -> KerasInputOutput:
        """
        Predicts using the model

        :param x: could be:
            1. A list of texts
            2. A pandas Series
            3. A pandas Dataframe
            4. A dict of lists containing texts
            5. A generator of any of the above formats
            6. A gianlp.utils.Sequence object that generates batches of text
        :param inference_batch: the prediction is made in batches for saving ram, this is the batch size used.
        ignored if x is a generator or a gianlp.utils.Sequence
        :param steps: steps for the generator, ignored if x is not a generator
        :param max_queue_size: Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
        :param workers: Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
        :param use_multiprocessing: If True, use process-based threading. If unspecified, use_multiprocessing will default to False. Note that because this implementation relies on multiprocessing, you should not pass non-picklable arguments to the generator as they can't be passed easily to children processes.
        :param verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = single line.
        :return: the output of the keras model
        """
        preds = None
        if isinstance(x, Sequence):
            steps = len(x) if not steps else min(steps, len(x))
            predict_generator = PredictSequenceWrapper(x, self.preprocess_texts)
            if use_multiprocessing:
                enq = OrderedEnqueuer(predict_generator, use_multiprocessing=True)
                enq.start(workers=workers, max_queue_size=max_queue_size)
                predict_generator = enq.get()
            else:
                predict_generator = predict_generator.__iter__()
        else:
            predict_generator = self._predict_generator(x, inference_batch)
            if use_multiprocessing and isinstance(x, types.GeneratorType):
                warning("Keras API allows prediction generators with multiprocessing, so does this method, "
                        "but be aware this completely looses track of which predictions are from which label "
                        "since order will be lost by concurrency. We recommend using a utils.Sequence object for"
                        " multiprocessing.")
                enq = GeneratorEnqueuer(predict_generator, use_multiprocessing=True,
                                        random_seed=self._random_seed)
                enq.start(workers=workers, max_queue_size=max_queue_size)
                predict_generator = enq.get()
            if not steps:
                if isinstance(x, types.GeneratorType):
                    raise ValueError("For using a generator the steps to use need to be specified.")
                x_len = self.__texts_input_length(x)
                steps = x_len // inference_batch + (1 if x_len % inference_batch != 0 else 0)
        for i in tqdm(range(steps), total=steps, disable=(True if verbose != 2 else False)):
            try:
                batch = next(predict_generator)
            except StopIteration:
                warning("Generator stopped before reaching the steps specified.")
                break
            pred_batch = self._get_keras_model().predict_on_batch(batch)
            preds = self.__merge_preds(preds, pred_batch)
            if verbose == 1:
                print(f"{i}/{steps} Batch predicted")
        if not isinstance(x, types.GeneratorType) and not isinstance(x, Sequence):
            if isinstance(preds, list):
                return [p[:self.__texts_input_length(x)] for p in preds]
            else:
                return preds[:self.__texts_input_length(x)]
        else:
            return preds

    def freeze(self) -> None:
        """
        Freezes the model weights
        :raises:
            ValueError:
            - When the model is not built
        """
        if not self._built:
            raise ValueError("Can't freeze a model that has not been built")
        model = self._get_keras_model()
        for k, v in model._get_trainable_state().items():
            k.trainable = False
        for inp in self._iterate_model_inputs(self.inputs):
            if isinstance(inp, TrainableModel):
                inp: TrainableModel
                inp.freeze()
