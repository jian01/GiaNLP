# Example: Drug rating regressor tutorial

We are going to use the Drug Review dataset. (Felix Gräßer, Surya Kallumadi, Hagen Malberg, and Sebastian Zaunseder. 2018. Aspect-Based Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning. In Proceedings of the 2018 International Conference on Digital Health (DH '18). ACM, New York, NY, USA, 121-125. DOI: [Web Link](https://doi.org/10.1145/3194658.3194677))


```python
!curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/00461/drugLib_raw.zip
!unzip drugLib_raw.zip
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 1106k  100 1106k    0     0   173k      0  0:00:06  0:00:06 --:--:--  224k
    Archive:  drugLib_raw.zip
      inflating: drugLibTest_raw.tsv     
      inflating: drugLibTrain_raw.tsv    



```python
import pandas as pd

train = pd.read_csv("drugLibTrain_raw.tsv", sep='\t')
test = pd.read_csv("drugLibTest_raw.tsv", sep='\t')
```


```python
print(train)
```

          Unnamed: 0       urlDrugName  rating           effectiveness  \
    0           2202         enalapril       4        Highly Effective   
    1           3117  ortho-tri-cyclen       1        Highly Effective   
    2           1146           ponstel      10        Highly Effective   
    3           3947          prilosec       3    Marginally Effective   
    4           1951            lyrica       2    Marginally Effective   
    ...          ...               ...     ...                     ...   
    3102        1039           vyvanse      10        Highly Effective   
    3103        3281            zoloft       1             Ineffective   
    3104        1664           climara       2    Marginally Effective   
    3105        2621         trileptal       8  Considerably Effective   
    3106        2748          micardis       4    Moderately Effective   
    
                            sideEffects                               condition  \
    0                 Mild Side Effects  management of congestive heart failure   
    1               Severe Side Effects                        birth prevention   
    2                   No Side Effects                        menstrual cramps   
    3                 Mild Side Effects                             acid reflux   
    4               Severe Side Effects                            fibromyalgia   
    ...                             ...                                     ...   
    3102              Mild Side Effects                                    adhd   
    3103  Extremely Severe Side Effects                              depression   
    3104          Moderate Side Effects                       total hysterctomy   
    3105              Mild Side Effects                                epilepsy   
    3106          Moderate Side Effects                     high blood pressure   
    
                                             benefitsReview  \
    0     slowed the progression of left ventricular dys...   
    1     Although this type of birth control has more c...   
    2     I was used to having cramps so badly that they...   
    3     The acid reflux went away for a few months aft...   
    4     I think that the Lyrica was starting to help w...   
    ...                                                 ...   
    3102  Increased focus, attention, productivity. Bett...   
    3103    Emotions were somewhat blunted. Less moodiness.   
    3104                                                ---   
    3105               Controlled complex partial seizures.   
    3106  The drug Micardis did seem to alleviate my hig...   
    
                                          sideEffectsReview  \
    0     cough, hypotension , proteinuria, impotence , ...   
    1     Heavy Cycle, Cramps, Hot Flashes, Fatigue, Lon...   
    2            Heavier bleeding and clotting than normal.   
    3     Constipation, dry mouth and some mild dizzines...   
    4     I felt extremely drugged and dopey.  Could not...   
    ...                                                 ...   
    3102  Restless legs at night, insomnia, headache (so...   
    3103  Weight gain, extreme tiredness during the day,...   
    3104  Constant issues with the patch not staying on....   
    3105                         Dizziness, fatigue, nausea   
    3106  I find when I am taking Micardis that I tend t...   
    
                                             commentsReview  
    0     monitor blood pressure , weight and asses for ...  
    1     I Hate This Birth Control, I Would Not Suggest...  
    2     I took 2 pills at the onset of my menstrual cr...  
    3     I was given Prilosec prescription at a dose of...  
    4                                             See above  
    ...                                                 ...  
    3102  I took adderall once as a child, and it made m...  
    3103  I was on Zoloft for about 2 years total. I am ...  
    3104                                                ---  
    3105  Started at 2 doses of 300 mg a day and worked ...  
    3106           I take Micardis in pill form once daily.  
    
    [3107 rows x 9 columns]


## Multi-text model building

The rating will be predicted using benefitsReview, commentsReview, sideEffectsReview and urlDrugName


```python
from gianlp.models import CharEmbeddingSequence, RNNDigest, KerasWrapper, PreTrainedWordEmbeddingSequence
```

    WARNING:nlp_builder:The NLP builder disables all tensorflow-related logging


### Word embedding for reviews
We are going to use a word embedding for the benefitsReview, commentsReview and sideEffectsReview. We first download uncased 50 dimensions glove word embedding, then we translate it to word2vec format.


```python
!curl -O http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
!unzip glove.6B
!python -m gensim.scripts.glove2word2vec --input  glove.6B.50d.txt --output glove.6B.50d.w2vformat.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
      0  822M    0 14216    0     0  29616      0  8:05:12 --:--:--  8:05:12 29555

We create our word embedding


```python
help(PreTrainedWordEmbeddingSequence.__init__)
```

    Help on function __init__ in module gianlp.models.text_representations.pre_trained_word_embedding:
    
    __init__(self, word2vec_src: Union[str, gensim.models.keyedvectors.Word2VecKeyedVectors], tokenizer: Callable[[str], List[str]], trainable: bool = False, sequence_maxlen: int = 20, random_state: int = 42)
        :param word2vec_src: path to word2vec format .txt file or gensim KeyedVectors
        :param tokenizer: a tokenizer function that transforms each string into a list of string tokens
                            the tokens transformed should match the keywords in the pretrained word embeddings
                            the function must support serialization through pickle
        :param trainable: if the vectors are trainable with keras backprop
        :param sequence_maxlen: The maximum allowed sequence length
        :param random_state: the random seed used for random processes
    


We will see the 80% percentile as a guideline for defining our `sequence_maxlen`.


```python
train['commentsReview'].fillna("").map(lambda x: len(x.split(" "))).quantile(0.8)
```
    76.0


With the tokenizer created, we can now create the :class:`.PreTrainedWordEmbeddingSequence`.

```python
def split_tokenizer(text):
    return text.lower().split(" ")

word_emb = PreTrainedWordEmbeddingSequence("glove.6B.50d.w2vformat.txt", 
                                           tokenizer=split_tokenizer, 
                                           sequence_maxlen=90)
```


```python
word_emb.outputs_shape
```

    (90, 50), float32



### Review digest

We build a different RNN for each review


```python
benefits_digest = RNNDigest(word_emb, units_per_layer=20, rnn_type='gru', stacked_layers=1)
comments_digest = RNNDigest(word_emb, units_per_layer=20, rnn_type='gru', stacked_layers=1)
side_eff_digest = RNNDigest(word_emb, units_per_layer=20, rnn_type='gru', stacked_layers=1)

benefits_digest.outputs_shape, comments_digest.outputs_shape, side_eff_digest.outputs_shape
```

    WARNING:nlp_builder:If the model and wrapper inputs mismatch it will only be noticed when building, before that output shape is an estimate and does not assert inputs.
    WARNING:nlp_builder:If the model and wrapper inputs mismatch it will only be noticed when building, before that output shape is an estimate and does not assert inputs.
    WARNING:nlp_builder:If the model and wrapper inputs mismatch it will only be noticed when building, before that output shape is an estimate and does not assert inputs.

    ((20,), float32, (20,), float32, (20,), float32)



### Char embedding for drug name

Since we know nothing about drug names I hope char embedding will help.


```python
char_emb = CharEmbeddingSequence(embedding_dimension=16, sequence_maxlen=30)
char_emb.outputs_shape
```

    (30, 16), float32



### Convolutional digest for drug name

We convolute the drug name to, hopefully, extract useful features.

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Input
```


```python
text_input = Input((30, 16))

conv_1char = Conv1D(5, kernel_size=1, padding="same")(text_input)
conv_2char = Conv1D(5, kernel_size=2, padding="same")(text_input)
conv_3char = Conv1D(5, kernel_size=3, padding="same")(text_input)
conv_4char = Conv1D(5, kernel_size=4, padding="same")(text_input)

state_seq = Concatenate()([conv_1char, conv_2char, conv_3char, conv_4char])
result = GlobalMaxPooling1D()(state_seq)

drug_encoder = Model(inputs=text_input, outputs=result)
drug_encoder.summary()
```

    Model: "model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_4 (InputLayer)           [(None, 30, 16)]     0           []                               
                                                                                                      
     conv1d (Conv1D)                (None, 30, 5)        85          ['input_4[0][0]']                
                                                                                                      
     conv1d_1 (Conv1D)              (None, 30, 5)        165         ['input_4[0][0]']                
                                                                                                      
     conv1d_2 (Conv1D)              (None, 30, 5)        245         ['input_4[0][0]']                
                                                                                                      
     conv1d_3 (Conv1D)              (None, 30, 5)        325         ['input_4[0][0]']                
                                                                                                      
     concatenate (Concatenate)      (None, 30, 20)       0           ['conv1d[0][0]',                 
                                                                      'conv1d_1[0][0]',               
                                                                      'conv1d_2[0][0]',               
                                                                      'conv1d_3[0][0]']               
                                                                                                      
     global_max_pooling1d (GlobalMa  (None, 20)          0           ['concatenate[0][0]']            
     xPooling1D)                                                                                      
                                                                                                      
    ==================================================================================================
    Total params: 820
    Trainable params: 820
    Non-trainable params: 0
    __________________________________________________________________________________________________



```python
drug_encoder = KerasWrapper(char_emb, drug_encoder)
print(drug_encoder)
```

    WARNING:nlp_builder:If the model and wrapper inputs mismatch it will only be noticed when building, before that output shape is an estimate and does not assert inputs.
            Model        |      Inputs shape     |      Output shape     |Trainable|  Total  |    Connected to    
                         |                       |                       | weights | weights |                    
    ==============================================================================================================
    7fcd9cdd9a60 CharEmbe|      (30,), int32     |   (30, 16), float32   |    ?    |    ?    |                    
    7fcd201327c0 KerasWra|   (30, 16), float32   |     (20,), float32    |    ?    |    ?    |7fcd9cdd9a60 CharEmb
    ==============================================================================================================
                         |                       |                       |    ?    |    ?    |                    


### Final model building

We have 20 dimensions for each review and 20 for the drug name, we have to build a regressor for that inputs


```python
drug_name = Input((20,))
review1 = Input((20,))
review2 = Input((20,))
review3 = Input((20,))

full_state = Concatenate()([drug_name, review1, review2, review3])
dense1 = Dense(50, activation='tanh')(full_state)
dense2 = Dense(50, activation='relu')(dense1)
dense3 = Dense(50, activation='relu')(dense2)
out = Dense(1, activation='relu')(dense3)

regressor = Model(inputs = [drug_name, review1, review2, review3], outputs = out)
```


```python
train.columns
```

    Index(['Unnamed: 0', 'urlDrugName', 'rating', 'effectiveness', 'sideEffects',
           'condition', 'benefitsReview', 'sideEffectsReview', 'commentsReview'],
          dtype='object')



We use multi-text for the inputs indicating we will be fitting different texts


```python
regressor = KerasWrapper([('urlDrugName',[drug_encoder]), 
                          ('benefitsReview', [benefits_digest]),
                         ('sideEffectsReview', [comments_digest]),
                         ('commentsReview', [side_eff_digest])], regressor)
```

### Model build


```python
regressor.build(train['commentsReview'].fillna("").tolist()+
                train['sideEffectsReview'].fillna("").tolist()+
                train['benefitsReview'].fillna("").tolist()+
                train['urlDrugName'].fillna("").tolist())
```


```python
print(regressor)
```

            Model        |      Inputs shape     |      Output shape     |Trainable|  Total  |    Connected to    
                         |                       |                       | weights | weights |                    
    ==============================================================================================================
    7fcda9e18c10 PreTrain|      (90,), int32     |   (90, 50), float32   |    0    | 20000100|                    
    7fcd20356460 KerasWra|   (90, 50), float32   |     (20,), float32    |   4320  | 20004420|7fcda9e18c10 PreTrai
    7fcd205f1430 KerasWra|   (90, 50), float32   |     (20,), float32    |   4320  | 20004420|7fcda9e18c10 PreTrai
    7fcd900b0700 KerasWra|   (90, 50), float32   |     (20,), float32    |   4320  | 20004420|7fcda9e18c10 PreTrai
    7fcd9cdd9a60 CharEmbe|      (30,), int32     |   (30, 16), float32   |   1456  |   1456  |                    
    7fcd201327c0 KerasWra|   (30, 16), float32   |     (20,), float32    |   2276  |   2276  |7fcd9cdd9a60 CharEmb
    7fcd205f5940 KerasWra|     (20,), float32    |     (1,), float32     |  24437  | 20024537|"urlDrugName": 7fcd2
                         |     (20,), float32    |                       |         |         |"benefitsReview": 7f
                         |     (20,), float32    |                       |         |         |"sideEffectsReview":
                         |     (20,), float32    |                       |         |         |"commentsReview": 7f
    ==============================================================================================================
                         |                       |                       |  24437  | 20024537|                    



```python
regressor.compile(optimizer="adam", loss="mse", metrics=["mae"])
```


```python
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20, monitor='val_loss', restore_best_weights=True)
```


```python
hst = regressor.fit(train, train['rating'].values,
                    batch_size=256, epochs=200, validation_split=0.1,
                    callbacks=[early_stopping])
```

    Epoch 1/200
    10/10 [==============================] - 15s 481ms/step - loss: 44.8526 - mae: 6.0467 - val_loss: 26.3905 - val_mae: 4.5891
    Epoch 2/200
    10/10 [==============================] - 2s 212ms/step - loss: 16.2550 - mae: 3.5820 - val_loss: 9.6836 - val_mae: 2.6122
    Epoch 3/200
    10/10 [==============================] - 2s 202ms/step - loss: 9.3684 - mae: 2.2984 - val_loss: 11.4970 - val_mae: 2.4750
    Epoch 4/200
    10/10 [==============================] - 2s 208ms/step - loss: 9.0430 - mae: 2.3599 - val_loss: 9.3396 - val_mae: 2.6122
    Epoch 5/200
    10/10 [==============================] - 2s 200ms/step - loss: 8.5684 - mae: 2.5149 - val_loss: 9.6518 - val_mae: 2.6384
    Epoch 6/200
    10/10 [==============================] - 2s 200ms/step - loss: 8.5556 - mae: 2.4241 - val_loss: 10.2140 - val_mae: 2.6156
    Epoch 7/200
    10/10 [==============================] - 2s 201ms/step - loss: 8.4898 - mae: 2.3640 - val_loss: 9.7481 - val_mae: 2.5665
    Epoch 8/200
    10/10 [==============================] - 2s 198ms/step - loss: 8.0966 - mae: 2.3293 - val_loss: 9.5756 - val_mae: 2.5541
    Epoch 9/200
    10/10 [==============================] - 2s 199ms/step - loss: 8.4019 - mae: 2.3864 - val_loss: 9.5679 - val_mae: 2.5755
    Epoch 10/200
    10/10 [==============================] - 2s 200ms/step - loss: 8.3462 - mae: 2.3844 - val_loss: 9.5266 - val_mae: 2.5618
    Epoch 11/200
    10/10 [==============================] - 2s 202ms/step - loss: 8.3780 - mae: 2.3694 - val_loss: 9.9598 - val_mae: 2.6336
    Epoch 12/200
    10/10 [==============================] - 2s 192ms/step - loss: 8.2200 - mae: 2.3633 - val_loss: 9.3986 - val_mae: 2.5457
    Epoch 13/200
    10/10 [==============================] - 2s 206ms/step - loss: 8.1503 - mae: 2.3485 - val_loss: 8.8225 - val_mae: 2.4505
    Epoch 14/200
    10/10 [==============================] - 2s 207ms/step - loss: 8.1233 - mae: 2.3438 - val_loss: 9.4460 - val_mae: 2.5943
    Epoch 15/200
    10/10 [==============================] - 2s 207ms/step - loss: 8.0737 - mae: 2.3433 - val_loss: 9.3705 - val_mae: 2.5378
    Epoch 16/200
    10/10 [==============================] - 2s 204ms/step - loss: 8.0723 - mae: 2.3258 - val_loss: 9.4884 - val_mae: 2.6086
    Epoch 17/200
    10/10 [==============================] - 2s 206ms/step - loss: 7.9773 - mae: 2.3296 - val_loss: 9.4471 - val_mae: 2.5593
    Epoch 18/200
    10/10 [==============================] - 2s 199ms/step - loss: 7.5464 - mae: 2.2537 - val_loss: 8.9789 - val_mae: 2.5062
    Epoch 19/200
    10/10 [==============================] - 2s 208ms/step - loss: 7.4574 - mae: 2.2392 - val_loss: 8.3895 - val_mae: 2.4031
    Epoch 20/200
    10/10 [==============================] - 2s 199ms/step - loss: 7.0000 - mae: 2.1540 - val_loss: 8.7761 - val_mae: 2.3631
    Epoch 21/200
    10/10 [==============================] - 2s 201ms/step - loss: 6.9204 - mae: 2.1558 - val_loss: 8.6919 - val_mae: 2.3652
    Epoch 22/200
    10/10 [==============================] - 2s 203ms/step - loss: 6.3791 - mae: 2.0272 - val_loss: 8.7601 - val_mae: 2.3497
    Epoch 23/200
    10/10 [==============================] - 2s 202ms/step - loss: 6.0366 - mae: 1.9443 - val_loss: 8.1706 - val_mae: 2.2950
    Epoch 24/200
    10/10 [==============================] - 2s 202ms/step - loss: 5.8741 - mae: 1.9132 - val_loss: 8.1911 - val_mae: 2.3053
    Epoch 25/200
    10/10 [==============================] - 2s 207ms/step - loss: 5.4787 - mae: 1.8285 - val_loss: 7.4014 - val_mae: 2.1681
    Epoch 26/200
    10/10 [==============================] - 2s 210ms/step - loss: 5.2076 - mae: 1.7880 - val_loss: 7.2368 - val_mae: 2.0988
    Epoch 27/200
    10/10 [==============================] - 2s 214ms/step - loss: 4.8116 - mae: 1.7061 - val_loss: 6.7278 - val_mae: 2.0793
    Epoch 28/200
    10/10 [==============================] - 2s 209ms/step - loss: 4.7443 - mae: 1.6859 - val_loss: 7.3081 - val_mae: 2.1378
    Epoch 29/200
    10/10 [==============================] - 2s 207ms/step - loss: 4.6793 - mae: 1.6791 - val_loss: 6.6134 - val_mae: 2.0334
    Epoch 30/200
    10/10 [==============================] - 2s 203ms/step - loss: 4.3276 - mae: 1.6048 - val_loss: 6.6267 - val_mae: 2.0093
    Epoch 31/200
    10/10 [==============================] - 2s 209ms/step - loss: 4.3379 - mae: 1.6148 - val_loss: 6.1516 - val_mae: 1.9605
    Epoch 32/200
    10/10 [==============================] - 2s 200ms/step - loss: 4.0363 - mae: 1.5443 - val_loss: 6.4251 - val_mae: 1.9917
    Epoch 33/200
    10/10 [==============================] - 2s 207ms/step - loss: 3.9923 - mae: 1.5460 - val_loss: 6.3441 - val_mae: 1.9595
    Epoch 34/200
    10/10 [==============================] - 2s 204ms/step - loss: 3.8146 - mae: 1.4986 - val_loss: 6.4183 - val_mae: 1.9715
    Epoch 35/200
    10/10 [==============================] - 2s 205ms/step - loss: 3.7252 - mae: 1.4755 - val_loss: 6.3719 - val_mae: 1.9541
    Epoch 36/200
    10/10 [==============================] - 2s 211ms/step - loss: 3.8593 - mae: 1.5039 - val_loss: 5.9912 - val_mae: 1.8993
    Epoch 37/200
    10/10 [==============================] - 2s 203ms/step - loss: 3.4515 - mae: 1.4152 - val_loss: 6.1809 - val_mae: 1.8888
    Epoch 38/200
    10/10 [==============================] - 2s 204ms/step - loss: 3.5217 - mae: 1.4361 - val_loss: 6.2674 - val_mae: 1.9094
    Epoch 39/200
    10/10 [==============================] - 2s 201ms/step - loss: 3.6698 - mae: 1.4673 - val_loss: 6.7534 - val_mae: 1.9594
    Epoch 40/200
    10/10 [==============================] - 2s 208ms/step - loss: 3.4128 - mae: 1.3989 - val_loss: 5.5831 - val_mae: 1.8298
    Epoch 41/200
    10/10 [==============================] - 2s 206ms/step - loss: 3.3113 - mae: 1.3933 - val_loss: 6.0933 - val_mae: 1.8533
    Epoch 42/200
    10/10 [==============================] - 2s 203ms/step - loss: 3.2604 - mae: 1.3622 - val_loss: 6.1341 - val_mae: 1.8971
    Epoch 43/200
    10/10 [==============================] - 2s 202ms/step - loss: 3.0881 - mae: 1.3521 - val_loss: 6.1046 - val_mae: 1.8776
    Epoch 44/200
    10/10 [==============================] - 2s 206ms/step - loss: 3.0806 - mae: 1.3335 - val_loss: 5.5088 - val_mae: 1.7957
    Epoch 45/200
    10/10 [==============================] - 2s 198ms/step - loss: 2.9925 - mae: 1.3121 - val_loss: 6.2356 - val_mae: 1.8951
    Epoch 46/200
    10/10 [==============================] - 2s 211ms/step - loss: 2.9967 - mae: 1.3252 - val_loss: 6.5368 - val_mae: 1.9281
    Epoch 47/200
    10/10 [==============================] - 2s 211ms/step - loss: 2.8649 - mae: 1.2917 - val_loss: 6.2750 - val_mae: 1.8579
    Epoch 48/200
    10/10 [==============================] - 2s 200ms/step - loss: 2.9601 - mae: 1.3048 - val_loss: 5.7847 - val_mae: 1.8140
    Epoch 49/200
    10/10 [==============================] - 2s 202ms/step - loss: 2.8342 - mae: 1.2702 - val_loss: 5.8610 - val_mae: 1.8357
    Epoch 50/200
    10/10 [==============================] - 2s 200ms/step - loss: 2.8497 - mae: 1.2969 - val_loss: 5.8004 - val_mae: 1.8159
    Epoch 51/200
    10/10 [==============================] - 2s 201ms/step - loss: 2.6426 - mae: 1.2303 - val_loss: 6.0414 - val_mae: 1.8613
    Epoch 52/200
    10/10 [==============================] - 2s 205ms/step - loss: 2.6518 - mae: 1.2339 - val_loss: 6.0281 - val_mae: 1.8042
    Epoch 53/200
    10/10 [==============================] - 2s 200ms/step - loss: 2.5120 - mae: 1.2305 - val_loss: 6.5550 - val_mae: 1.9098
    Epoch 54/200
    10/10 [==============================] - 2s 201ms/step - loss: 2.5509 - mae: 1.2092 - val_loss: 6.2900 - val_mae: 1.8720
    Epoch 55/200
    10/10 [==============================] - 2s 204ms/step - loss: 2.4135 - mae: 1.1927 - val_loss: 5.6932 - val_mae: 1.7901
    Epoch 56/200
    10/10 [==============================] - 2s 200ms/step - loss: 2.3593 - mae: 1.1683 - val_loss: 5.9240 - val_mae: 1.7881
    Epoch 57/200
    10/10 [==============================] - 2s 204ms/step - loss: 2.3442 - mae: 1.1779 - val_loss: 6.2036 - val_mae: 1.8304
    Epoch 58/200
    10/10 [==============================] - 2s 204ms/step - loss: 2.2430 - mae: 1.1506 - val_loss: 6.2946 - val_mae: 1.8458
    Epoch 59/200
    10/10 [==============================] - 2s 203ms/step - loss: 2.2977 - mae: 1.1396 - val_loss: 5.9096 - val_mae: 1.8211
    Epoch 60/200
    10/10 [==============================] - 2s 205ms/step - loss: 2.1639 - mae: 1.1418 - val_loss: 5.5687 - val_mae: 1.7562
    Epoch 61/200
    10/10 [==============================] - 2s 207ms/step - loss: 2.0508 - mae: 1.1006 - val_loss: 5.4898 - val_mae: 1.7547
    Epoch 62/200
    10/10 [==============================] - 2s 209ms/step - loss: 2.1567 - mae: 1.1206 - val_loss: 5.9577 - val_mae: 1.8105
    Epoch 63/200
    10/10 [==============================] - 2s 203ms/step - loss: 2.0232 - mae: 1.0940 - val_loss: 6.1129 - val_mae: 1.8515
    Epoch 64/200
    10/10 [==============================] - 2s 207ms/step - loss: 2.0656 - mae: 1.0940 - val_loss: 6.0248 - val_mae: 1.8243
    Epoch 65/200
    10/10 [==============================] - 2s 202ms/step - loss: 1.8792 - mae: 1.0665 - val_loss: 5.8070 - val_mae: 1.7824
    Epoch 66/200
    10/10 [==============================] - 2s 202ms/step - loss: 1.9632 - mae: 1.0711 - val_loss: 5.9124 - val_mae: 1.7691
    Epoch 67/200
    10/10 [==============================] - 2s 194ms/step - loss: 2.0155 - mae: 1.0864 - val_loss: 6.1221 - val_mae: 1.8192
    Epoch 68/200
    10/10 [==============================] - 2s 201ms/step - loss: 1.8581 - mae: 1.0461 - val_loss: 6.1425 - val_mae: 1.8245
    Epoch 69/200
    10/10 [==============================] - 2s 204ms/step - loss: 1.8283 - mae: 1.0437 - val_loss: 6.1746 - val_mae: 1.8088
    Epoch 70/200
    10/10 [==============================] - 2s 204ms/step - loss: 1.7469 - mae: 1.0271 - val_loss: 6.1359 - val_mae: 1.8005
    Epoch 71/200
    10/10 [==============================] - 2s 199ms/step - loss: 1.7665 - mae: 1.0200 - val_loss: 6.1712 - val_mae: 1.7875
    Epoch 72/200
    10/10 [==============================] - 2s 204ms/step - loss: 1.7146 - mae: 1.0164 - val_loss: 6.4598 - val_mae: 1.8229
    Epoch 73/200
    10/10 [==============================] - 2s 202ms/step - loss: 1.7306 - mae: 1.0091 - val_loss: 6.0780 - val_mae: 1.7763
    Epoch 74/200
    10/10 [==============================] - 2s 199ms/step - loss: 1.6304 - mae: 0.9804 - val_loss: 5.8898 - val_mae: 1.7796
    Epoch 75/200
    10/10 [==============================] - 2s 201ms/step - loss: 1.5902 - mae: 0.9769 - val_loss: 5.8832 - val_mae: 1.7269
    Epoch 76/200
    10/10 [==============================] - 2s 202ms/step - loss: 1.5511 - mae: 0.9606 - val_loss: 6.5852 - val_mae: 1.8586
    Epoch 77/200
    10/10 [==============================] - 2s 208ms/step - loss: 1.5500 - mae: 0.9463 - val_loss: 6.2416 - val_mae: 1.8236
    Epoch 78/200
    10/10 [==============================] - 2s 199ms/step - loss: 1.5069 - mae: 0.9439 - val_loss: 6.6130 - val_mae: 1.9194
    Epoch 79/200
    10/10 [==============================] - 2s 207ms/step - loss: 1.4580 - mae: 0.9301 - val_loss: 6.3010 - val_mae: 1.8422
    Epoch 80/200
    10/10 [==============================] - 2s 202ms/step - loss: 1.4190 - mae: 0.9184 - val_loss: 5.8961 - val_mae: 1.7574
    Epoch 81/200
    10/10 [==============================] - 2s 207ms/step - loss: 1.4412 - mae: 0.9182 - val_loss: 6.0843 - val_mae: 1.7733


## Evaluation


```python
preds = regressor.predict(test)
```


```python
preds
```

    array([[7.9903536],
           [9.336563 ],
           [1.3078178],
           ...,
           [8.757404 ],
           [9.284457 ],
           [8.320618 ]], dtype=float32)




```python
preds.min(), preds.max()
```

    (0.0, 10.139272)




```python
print("MAE: ", abs(test['rating'].values-preds.flatten()).mean())
```

    MAE:  1.8476456113691841


## Serialization and deserialization


```python
from gianlp.models import BaseModel

data = regressor.serialize()
regressor2 = BaseModel.deserialize(data)
regressor2.predict(test)
```

    array([[7.9903536],
           [9.336563 ],
           [1.3078178],
           ...,
           [8.757404 ],
           [9.284457 ],
           [8.320618 ]], dtype=float32)



```python
print(regressor2)
```

            Model        |      Inputs shape     |      Output shape     |Trainable|  Total  |    Connected to    
                         |                       |                       | weights | weights |                    
    ==============================================================================================================
    7fcbfb9dbdc0 PreTrain|      (90,), int32     |   (90, 50), float32   |    0    | 20000100|                    
    7fcc095f1130 KerasWra|   (90, 50), float32   |     (20,), float32    |   4320  | 20004420|7fcbfb9dbdc0 PreTrai
    7fccada16100 PreTrain|      (90,), int32     |   (90, 50), float32   |    0    | 20000100|                    
    7fccae14c8e0 KerasWra|   (90, 50), float32   |     (20,), float32    |   4320  | 20004420|7fccada16100 PreTrai
    7fccd117e280 PreTrain|      (90,), int32     |   (90, 50), float32   |    0    | 20000100|                    
    7fcd2065d160 KerasWra|   (90, 50), float32   |     (20,), float32    |   4320  | 20004420|7fccd117e280 PreTrai
    7fccd3688280 CharEmbe|      (30,), int32     |   (30, 16), float32   |   1456  |   1456  |                    
    7fccd3688f40 KerasWra|   (30, 16), float32   |     (20,), float32    |   2276  |   2276  |7fccd3688280 CharEmb
    7fcd2065d250 KerasWra|     (20,), float32    |     (1,), float32     |  24437  | 60024737|"urlDrugName": 7fccd
                         |     (20,), float32    |                       |         |         |"benefitsReview": 7f
                         |     (20,), float32    |                       |         |         |"sideEffectsReview":
                         |     (20,), float32    |                       |         |         |"commentsReview": 7f
    ==============================================================================================================
                         |                       |                       |  24437  | 60024737|                    



