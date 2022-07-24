# Quickstart: Binary Classifier Tutorial

We are going to build a binary classifier for the [SMS Spam collection](https://www.dt.fee.unicamp.br/~tiago/smsspamcollection/). We start by downloading it.


```python
!curl -O https://raw.githubusercontent.com/justmarkham/DAT5/master/data/SMSSpamCollection.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  466k  100  466k    0     0   944k      0 --:--:-- --:--:-- --:--:--  946k



```python
import pandas as pd
```


```python
dataset = pd.read_csv('SMSSpamCollection.txt', sep='\t', header=None, names=['label', 'text'])
print(dataset.sample(5))
```

         label                                               text
    4898   ham  I cant pick the phone right now. Pls send a me...
    3023   ham                        How dare you change my ring
    333   spam  Call Germany for only 1 pence per minute! Call...
    3934   ham                             Playin space poker, u?
    12    spam  URGENT! You have won a 1 week FREE membership ...


Across tutorials we are going to build different classifier architectures, we will start with the simpler one: char embedding sequence

## Char embedding sequence classifier


```python
from gianlp.models import CharEmbeddingSequence, RNNDigest, KerasWrapper
```

    WARNING:nlp_builder:The NLP builder disables all tensorflow-related logging


We create a char embedding sequences for the texts, with a dimension for each char embedding of 32 and a sequence maxlen that matches percentile 80 of texts lengths

### Text representations


```python
help(CharEmbeddingSequence.__init__)
```
    Help on function __init__ in module gianlp.models.text_representations.char_embedding_sequence:
    
    __init__(self, embedding_dimension: int = 256, sequence_maxlen: int = 80, min_freq_percentile: int = 5, random_state: int = 42)
        :param embedding_dimension: The char embedding dimension
        :param sequence_maxlen: The maximum allowed sequence length
        :param min_freq_percentile: minimum percentile of the frequency for keeping a char.
                                    If a char has a frequency lower than this percentile it
                                    would be treated as unknown.
        :param random_state: random seed

```python
char_emb = CharEmbeddingSequence(embedding_dimension=32, sequence_maxlen=dataset['text'].str.len().quantile(0.8))
```

We can see that the output shape of the char embedding is a sequence of at most 137 chars with 32 dimensions each


```python
char_emb.outputs_shape
```
    (137, 32), float32



We also have an input shape for interacting with keras models


```python
char_emb.inputs_shape
```
    (137,), int32



### Sequence digest
The output sequence will be colapsed in a single state using RNNs


```python
help(RNNDigest.__new__)
```
    Help on function __new__ in module gianlp.models.rnn_digest:
    
    __new__(cls, inputs: Union[List[ForwardRef('BaseModel')], List[Tuple[str, List[ForwardRef('BaseModel')]]], gianlp.models.base_model.BaseModel], units_per_layer: int, rnn_type: str, stacked_layers: int = 1, masking: bool = True, bidirectional: bool = False, random_seed: int = 42, **kwargs)
        :param inputs: the inputs of the model
        :param units_per_layer: the amount of units per layer
        :param rnn_type: the type of rnn, could be "rnn", "gru" or "lstm"
        :param stacked_layers: the amount of layers to stack, 1 by default
        :param masking: if apply masking with 0 to the sequence
        :param bidirectional: if it's bidirectional
        :param random_seed: the seed for random processes
        :param kwargs: extra arguments for the rnn layers
    



```python
rnn_digest = RNNDigest(char_emb, units_per_layer=40, rnn_type='gru', stacked_layers=2)
```

The output shape is now a single vector


```python
rnn_digest.outputs_shape
```

```
    WARNING:nlp_builder:If the model and wrapper inputs mismatch it will only be noticed when building, before that output shape is an estimate and does not assert inputs.
    (40,), float32
```


### Binary classifier

Now we just simply build a binary classifier with an input of 40 floats and compile the keras model as always


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```


```python
model = Sequential()

model.add(Dense(20, input_shape=(40,), activation='tanh'))
model.add(Dense(20, activation='tanh'))
model.add(Dense(20, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
```


```python
help(KerasWrapper.__init__)
```
    Help on function __init__ in module gianlp.models.keras_wrapper:
    
    __init__(self, inputs: Union[List[ForwardRef('BaseModel')], List[Tuple[str, List[ForwardRef('BaseModel')]]], gianlp.models.base_model.BaseModel], wrapped_model: keras.engine.training.Model, **kwargs)
        :param inputs: the models that are the input of this one. Either a list containing model inputs one by one or a
        dict indicating which text name is assigned to which inputs.
        If a list, all should have multi-text input or don't have it. If it's a dict all shouldn't have multi-text
        input.
        :param wrapped_model: the keras model to wrap.
                                if it has multiple inputs, inputs should be a list and have the same len
        :param random_seed: random seed used in training
        :raises:
            ValueError:
            - When the wrapped model is not a keras model
            - When the keras model to wrap does not have a defined input shape
            - When inputs is a list of models and some of the models in the input have multi-text input and others
            don't.
            - When inputs is a list of tuples and any of the models has multi-text input.
            - When inputs is a list of tuples with length one
            - When inputs is a list containing some tuples of (str, model) and some models
            - When the wrapped model has multiple inputs and the inputs don't have the same length as the inputs in
            wrapped model
    



```python
model = KerasWrapper(rnn_digest, model)
model.outputs_shape
```
    WARNING:nlp_builder:If the model and wrapper inputs mismatch it will only be noticed when building, before that output shape is an estimate and does not assert inputs.
    WARNING:nlp_builder:If the model and wrapper inputs mismatch it will only be noticed when building, before that output shape is an estimate and does not assert inputs.

    (1,), float32

We can see a model summary


```python
print(model)
```
    WARNING:nlp_builder:If the model and wrapper inputs mismatch it will only be noticed when building, before that output shape is an estimate and does not assert inputs.
    WARNING:nlp_builder:If the model and wrapper inputs mismatch it will only be noticed when building, before that output shape is an estimate and does not assert inputs.
    WARNING:nlp_builder:If the model and wrapper inputs mismatch it will only be noticed when building, before that output shape is an estimate and does not assert inputs.
    WARNING:nlp_builder:If the model and wrapper inputs mismatch it will only be noticed when building, before that output shape is an estimate and does not assert inputs.
            Model        |      Inputs shape     |      Output shape     |Trainable|  Total  |    Connected to    
                         |                       |                       | weights | weights |                    
    ==============================================================================================================
    7fd2a8539c70 CharEmbe|     (137,), int32     |   (137, 32), float32  |    ?    |    ?    |                    
    7fd2300e7520 KerasWra|   (137, 32), float32  |     (40,), float32    |    ?    |    ?    |7fd2a8539c70 CharEmb
    7fd1e0413520 KerasWra|     (40,), float32    |     (1,), float32     |    ?    |    ?    |7fd2300e7520 KerasWr
    ==============================================================================================================
                         |                       |                       |    ?    |    ?    |                    


Note all the warnings and the `?` symbols at the weights. This is because the model has not yet been built, the numbers shown will only work if the models are connected properly and the weights are impossible to know if the representations are not built.

### Train-test split


```python
dataset = dataset.sample(len(dataset))
train = dataset.iloc[:int(len(dataset)*0.8)]
test = dataset.iloc[-int(len(dataset)*0.8):]
```

Our models need to be built always with a corpus of texts for the text representations to learn how to preprocess. In the case of the char embedding it needs to learn the most common chars to know how many vector will train.


```python
model.build(train['text'])
```

We can see the complete summary now


```python
print(model)
```
            Model        |      Inputs shape     |      Output shape     |Trainable|  Total  |    Connected to    
                         |                       |                       | weights | weights |                    
    ==============================================================================================================
    7fd2a8539c70 CharEmbe|     (137,), int32     |   (137, 32), float32  |   3392  |   3392  |                    
    7fd2300e7520 KerasWra|   (137, 32), float32  |     (40,), float32    |  22112  |  22112  |7fd2a8539c70 CharEmb
    7fd1e0413520 KerasWra|     (40,), float32    |     (1,), float32     |  23793  |  23793  |7fd2300e7520 KerasWr
    ==============================================================================================================
                         |                       |                       |  23793  |  23793  |                    


### Training


```python
type(model)
```




    gianlp.models.keras_wrapper.KerasWrapper




```python
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```


```python
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=2, mode='max', monitor='val_accuracy', restore_best_weights=True)
```


```python
help(model.fit)
```
    Help on method fit in module gianlp.models.trainable_model:
    
    fit(x: Union[Generator[Tuple[Union[List[str], pandas.core.series.Series, Dict[str, List[str]], pandas.core.frame.DataFrame], Union[List[<built-in function array>], <built-in function array>]], NoneType, NoneType], List[str], pandas.core.series.Series, Dict[str, List[str]], pandas.core.frame.DataFrame] = None, y: Optional[<built-in function array>] = None, batch_size: int = 32, epochs: int = 1, verbose: Union[str, int] = 'auto', callbacks: List[keras.callbacks.Callback] = None, validation_split: Optional[float] = 0.0, validation_data: Union[Generator[Tuple[Union[List[str], pandas.core.series.Series, Dict[str, List[str]], pandas.core.frame.DataFrame], Union[List[<built-in function array>], <built-in function array>]], NoneType, NoneType], Tuple[Union[List[str], pandas.core.series.Series, Dict[str, List[str]], pandas.core.frame.DataFrame], Union[List[<built-in function array>], <built-in function array>]], NoneType] = None, steps_per_epoch: Optional[int] = None, validation_steps: Optional[int] = None) -> keras.callbacks.History method of gianlp.models.keras_wrapper.KerasWrapper instance
        Fits the model
        
        :param x: Input data. Could be:
            1. A generator that yields (x, y) where x is any valid format for x and y is the target numpy array
            2. A list of texts
            3. A pandas Series
            4. A pandas Dataframe
            5. A dict of lists containing texts
        :param y: Target, ignored if x is a generator. Numpy array.
        :param batch_size: Batch size for training, ignored if x is a generator
        :param epochs: Amount of epochs to train
        :param verbose: verbose mode for Keras training
        :param callbacks: list of Callback objects for Keras model
        :param validation_split: the proportion of data to use for validation, ignored if x is a generator
        :param validation_data: Validation data. Could be:
            1. A tuple containing (x, y) where x is a list of text and y is the target numpy array
            2. A generator that yields (x, y) where x is a list of texts and y is the target numpy array
        :param steps_per_epoch: Amount of generator steps to consider an epoch as finished. Ignored if x is not a
        generator
        :param validation_steps: Amount of generator steps to consider to feed each validation evaluation.
                                Ignored if validation_data is not a generator
        :return: A History object. Its History.history attribute is a record of training loss values and metrics values
        at successive epochs, as well as validation loss values and validation metrics values (if applicable).
    



```python
hst = model.fit(train['text'], train['label'].map(lambda x: 1 if x=='spam' else 0).values,
                batch_size=256, epochs=30, validation_split=0.1,
                callbacks=[early_stopping])
```
    Epoch 1/30
    15/15 [==============================] - 10s 162ms/step - loss: 0.4862 - accuracy: 0.8016 - val_loss: 0.3075 - val_accuracy: 0.8828
    Epoch 2/30
    15/15 [==============================] - 1s 35ms/step - loss: 0.3187 - accuracy: 0.8805 - val_loss: 0.2899 - val_accuracy: 0.8906
    Epoch 3/30
    15/15 [==============================] - 0s 32ms/step - loss: 0.2495 - accuracy: 0.9112 - val_loss: 0.2561 - val_accuracy: 0.9102
    Epoch 4/30
    15/15 [==============================] - 0s 32ms/step - loss: 0.2233 - accuracy: 0.9245 - val_loss: 0.1870 - val_accuracy: 0.9336
    Epoch 5/30
    15/15 [==============================] - 0s 31ms/step - loss: 0.1731 - accuracy: 0.9430 - val_loss: 0.1805 - val_accuracy: 0.9414
    Epoch 6/30
    15/15 [==============================] - 0s 32ms/step - loss: 0.1335 - accuracy: 0.9589 - val_loss: 0.1678 - val_accuracy: 0.9609
    Epoch 7/30
    15/15 [==============================] - 0s 31ms/step - loss: 0.1227 - accuracy: 0.9651 - val_loss: 0.1681 - val_accuracy: 0.9453
    Epoch 8/30
    15/15 [==============================] - 0s 33ms/step - loss: 0.0929 - accuracy: 0.9750 - val_loss: 0.1784 - val_accuracy: 0.9531


### Testing


```python
help(model.predict)
```
    Help on method predict in module gianlp.models.trainable_model:
    
    predict(x: Union[Generator[Union[List[str], pandas.core.series.Series, Dict[str, List[str]], pandas.core.frame.DataFrame], NoneType, NoneType], List[str], pandas.core.series.Series, Dict[str, List[str]], pandas.core.frame.DataFrame], steps: Optional[int] = None, **kwargs: Any) -> Union[List[<built-in function array>], <built-in function array>] method of gianlp.models.keras_wrapper.KerasWrapper instance
        Predicts using the model
        
        :param x: could be:
            1. A list of texts
            2. A pandas Series
            3. A pandas Dataframe
            4. A dict of lists containing texts
            5. A generator of any of the above formats
        :param steps: steps for the generator, ignored if x is a list
        :param kwargs: arguments for keras predict method
        :return: the output of the keras model
    



```python
test_preds = model.predict(test['text'])
```


```python
test_preds
```
    array([[0.01580496],
           [0.01534021],
           [0.9671472 ],
           ...,
           [0.01837054],
           [0.01742908],
           [0.01805047]], dtype=float32)

```python
import numpy as np

print("Accuracy: ",
      np.equal((test['label'].map(lambda x: 1 if x=='spam' else 0).values==1), test_preds.flatten()>0.5).sum()/len(test))
```
    Accuracy:  0.9614090195198564


### Serialization

The model is serialized as bytes for loading later

```python
data = model.serialize()
type(data), len(data)
```
    (bytes, 661113)




```python
from gianlp.models import BaseModel
model2 = BaseModel.deserialize(data)
```

When serialized the model lost some of our own library layers, this is optional and is meant to simplify the graph and make the deserialization faster.


```python
print(model2)
```
            Model        |      Inputs shape     |      Output shape     |Trainable|  Total  |    Connected to    
                         |                       |                       | weights | weights |                    
    ==============================================================================================================
    7fd180ab8b20 CharEmbe|     (137,), int32     |   (137, 32), float32  |   3392  |   3392  |                    
    7fd15bd90190 KerasWra|   (137, 32), float32  |     (1,), float32     |  23793  |  23793  |7fd180ab8b20 CharEmb
    ==============================================================================================================
                         |                       |                       |  23793  |  23793  |                    

```python
test_preds2 = model2.predict(test['text'])
```

```python
assert test_preds.flatten().tolist() == test_preds2.flatten().tolist()
```