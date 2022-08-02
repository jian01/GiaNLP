# Advanced example: Ner Tagger via BiLSTM-CRF

We are going to use the famous CoNLL-2003 dataset (Sang, E. F., & De Meulder, F. (2003). Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. arXiv preprint cs/0306050.)

For the NER tagger we will use an architecture inspired by the BiLSTM-CRF for sequences paper (Huang, Z., Xu, W., & Yu, K. (2015). Bidirectional LSTM-CRF models for sequence tagging. arXiv preprint arXiv:1508.01991.)

This is an advanced usage and is a clear example of what kind of architectures/handling of outputs/losses the library CAN'T support from end to end, but can be of help.


```python
!curl -OL https://data.deepai.org/conll2003.zip 
!unzip conll2003.zip
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  959k  100  959k    0     0   387k      0  0:00:02  0:00:02 --:--:--  387k
    Archive:  conll2003.zip
      inflating: metadata                
      inflating: test.txt                
      inflating: train.txt               
      inflating: valid.txt               



```python
def load_data_file(filename):
    with open(filename, 'r') as file:
        data = file.read()
    data = data.split('\n\n')[1:] # remove header
    texts = [[w_tuple.split(' ')[0] for w_tuple in text.split('\n')] for text in data]
    tags = [[w_tuple.split(' ')[-1] for w_tuple in text.split('\n')] for text in data]
    return texts, tags
```


```python
train_texts, train_tags = load_data_file('train.txt')
```


```python
valid_texts, valid_tags = load_data_file('valid.txt')
```


```python
test_texts, test_tags = load_data_file('test.txt')
```

## Model architecture

We are going to use a word embedding and a char embedding per word.

### Word embeddings


```python
!curl -O http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
!unzip -o glove.6B
!python -m gensim.scripts.glove2word2vec --input  glove.6B.50d.txt --output glove.6B.50d.w2vformat.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  822M  100  822M    0     0  5172k      0  0:02:42  0:02:42 --:--:-- 5486k2:34  0:00:19  0:02:15 5122k
    Archive:  glove.6B.zip
    replace glove.6B.50d.txt? [y]es, [n]o, [A]ll, [N]one, [r]ename: ^C
    /home/jian01/.pyenv/versions/3.9.5/lib/python3.9/runpy.py:127: RuntimeWarning: 'gensim.scripts.glove2word2vec' found in sys.modules after import of package 'gensim.scripts', but prior to execution of 'gensim.scripts.glove2word2vec'; this may result in unpredictable behaviour
      warn(RuntimeWarning(msg))
    2022-02-21 23:22:04,579 - glove2word2vec - INFO - running /home/jian01/.pyenv/versions/meli/lib/python3.9/site-packages/gensim/scripts/glove2word2vec.py --input glove.6B.50d.txt --output glove.6B.50d.w2vformat.txt
    2022-02-21 23:22:08,057 - glove2word2vec - INFO - converting 400000 vectors from glove.6B.50d.txt to glove.6B.50d.w2vformat.txt
    2022-02-21 23:22:15,736 - glove2word2vec - INFO - Converted model with 400000 vectors and 50 dimensions



We don't need to tokenize


```python
def dummy_tokenizer(x):
    return x

def lower_dummy_tokenizer(x):
    return [w.lower() for w in x]
```


```python
max([len(t) for t in train_tags])
```

    113




```python
test_texts = [t[:113] for t in test_texts]
valid_texts = [t[:113] for t in valid_texts]
```


```python
word_emb = PreTrainedWordEmbeddingSequence("glove.6B.50d.w2vformat.txt", 
                                           tokenizer=lower_dummy_tokenizer, 
                                           sequence_maxlen=113)
```


```python
word_emb.outputs_shape
```

    (113, 50), float32



### Char embedding per word followed by digest


```python
char_per_word = CharPerWordEmbeddingSequence(tokenizer=dummy_tokenizer, embedding_dimension=32, 
                                             word_maxlen=113, char_maxlen=10)
char_digest_per_word = RNNDigest(char_per_word, units_per_layer=30, rnn_type='gru', stacked_layers=3)
```


```python
char_digest_per_word.outputs_shape
```

    WARNING:nlp_builder:If the model and wrapper inputs mismatch it will only be noticed when building, before that output shape is an estimate and does not assert inputs.

    (113, 30), float32



### BiLSTM-CRF

We can't build the BiLSTM-CRF using the library because the loss if very hard to set, so we are going to use the library up to this point and continue using keras


```python
!pip install tensorflow-addons
```

    Requirement already satisfied: tensorflow-addons in /home/jian01/.pyenv/versions/3.9.5/envs/meli/lib/python3.9/site-packages (0.16.1)
    Requirement already satisfied: typeguard>=2.7 in /home/jian01/.pyenv/versions/3.9.5/envs/meli/lib/python3.9/site-packages (from tensorflow-addons) (2.13.3)
    [33mWARNING: You are using pip version 21.1.1; however, version 22.0.3 is available.
    You should consider upgrading via the '/home/jian01/.pyenv/versions/3.9.5/envs/meli/bin/python3.9 -m pip install --upgrade pip' command.[0m


We need to know the amount of labels


```python
possible_tags = list(set([tag for text_tags in train_tags for tag in text_tags if tag]))
possible_tags
```

    ['I-PER', 'I-MISC', 'I-ORG', 'B-PER', 'B-ORG', 'O', 'I-LOC', 'B-MISC', 'B-LOC']




```python
from tensorflow.keras.layers import Dense, Conv1D, Masking, Concatenate, Input, LSTM, Bidirectional, Lambda, Flatten
from tensorflow.keras.models import Model, Sequential
```

We will use a custom CRF layer taken from https://github.com/Damcy/mytf2/blob/59135d91b57f041029885ebff51f4119c2aa1677/mytf2/layer/CRF.py

It is extremely hard to use the CRF layer from tensorflow addons along with keras since it has 4 outputs of different shapes and computing a loss for that inside keras is not trivial. A CRF is a probabilistic model involving multiple parameters so all must be considered in the likelihood calculation.


```python
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow_addons.text import crf_log_likelihood

class CRF(layers.Layer):
    def __init__(self, label_size):
        super(CRF, self).__init__()
        self.trans_params = tf.Variable(
            tf.random.uniform(shape=(label_size, label_size)), name='transition')
    
    @tf.function
    def call(self, inputs, labels, seq_lens):
        inputs = tf.convert_to_tensor(inputs)
        log_likelihood, self.trans_params = crf_log_likelihood(
                                                inputs, labels, seq_lens,
                                                transition_params=self.trans_params)
        loss = tf.reduce_sum(-log_likelihood)
        return loss
```

The transition potentials are modeled in the CRF and the unary potentials are modelled by the dense layers


```python
inp1 = Input((113,50), name='inp1')
inp2 = Input((113,30), name='inp2')
targets = Input(shape=(113,), name='target_ids', dtype='int32')
seq_lens = Input(shape=(), name='input_lens', dtype='int32')
inp1 = Masking(0.0)(inp1)
x = Concatenate()([inp1, inp2])
x = Bidirectional(LSTM(20, return_sequences=True))(x)
x = Bidirectional(LSTM(20, return_sequences=True))(x)
x = Bidirectional(LSTM(20, return_sequences=True))(x)
x = Bidirectional(LSTM(20, return_sequences=True))(x)
x = Bidirectional(LSTM(20, return_sequences=True))(x)
unary = Dense(len(possible_tags), activation='tanh')(x)
unary = Dense(len(possible_tags), activation='tanh')(unary)
unary = Dense(len(possible_tags), activation='tanh')(unary)
logits = Dense(len(possible_tags), activation='tanh', name='logits')(unary)
loss = CRF(len(possible_tags))(logits, targets, seq_lens)

model = Model(inputs=[inp1, inp2, targets, seq_lens], outputs=loss)

model.summary()
```

    Model: "model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_2 (InputLayer)           [(None, 113, 50)]    0           []                               
                                                                                                      
     inp2 (InputLayer)              [(None, 113, 30)]    0           []                               
                                                                                                      
     concatenate (Concatenate)      (None, 113, 80)      0           ['input_2[0][0]',                
                                                                      'inp2[0][0]']                   
                                                                                                      
     bidirectional (Bidirectional)  (None, 113, 40)      16160       ['concatenate[1][0]']            
                                                                                                      
     bidirectional_1 (Bidirectional  (None, 113, 40)     9760        ['bidirectional[1][0]']          
     )                                                                                                
                                                                                                      
     bidirectional_2 (Bidirectional  (None, 113, 40)     9760        ['bidirectional_1[1][0]']        
     )                                                                                                
                                                                                                      
     bidirectional_3 (Bidirectional  (None, 113, 40)     9760        ['bidirectional_2[1][0]']        
     )                                                                                                
                                                                                                      
     bidirectional_4 (Bidirectional  (None, 113, 40)     9760        ['bidirectional_3[1][0]']        
     )                                                                                                
                                                                                                      
     dense (Dense)                  (None, 113, 9)       369         ['bidirectional_4[1][0]']        
                                                                                                      
     dense_1 (Dense)                (None, 113, 9)       90          ['dense[1][0]']                  
                                                                                                      
     dense_2 (Dense)                (None, 113, 9)       90          ['dense_1[1][0]']                
                                                                                                      
     logits (Dense)                 (None, 113, 9)       90          ['dense_2[1][0]']                
                                                                                                      
     target_ids (InputLayer)        [(None, 113)]        0           []                               
                                                                                                      
     input_lens (InputLayer)        [(None,)]            0           []                               
                                                                                                      
     crf (CRF)                      ()                   81          ['logits[1][0]',                 
                                                                      'target_ids[0][0]',             
                                                                      'input_lens[0][0]']             
                                                                                                      
    ==================================================================================================
    Total params: 55,920
    Trainable params: 55,920
    Non-trainable params: 0
    __________________________________________________________________________________________________


## Model Building


```python
char_digest_per_word.build([" ".join(text) for text in train_texts])
word_emb.build([" ".join(text) for text in train_texts])
```

## Training


```python
import numpy as np
from tensorflow.keras.preprocessing import sequence as keras_seq

def preprocess_tags(tags, possible_tags, maxlen):
    new_seqs = []
    sequence_lens = []
    for tag_seq in tags:
        new_seq = []
        for t in tag_seq:
            if t in possible_tags:
                new_seq.append(possible_tags.index(t))
            else:
                new_seq.append(possible_tags.index('O'))
        new_seqs.append(new_seq)
        sequence_lens.append(len(new_seq))
    return keras_seq.pad_sequences(new_seqs, maxlen=maxlen, dtype="int32",
                                   padding="post", truncating="post",value=-1), np.asarray(sequence_lens)
```


```python
train_tags_prep, train_lens = preprocess_tags(train_tags, possible_tags, 113)
test_tags_prep, test_lens = preprocess_tags(test_tags, possible_tags, 113)
valid_tags_prep, valid_lens = preprocess_tags(valid_tags, possible_tags, 113)
```


```python
train_tags_prep.shape
```

    (14987, 113)




```python
train_tags_prep[0], train_lens[0]
```

    (array([ 4,  5,  7,  5,  5,  5,  7,  5,  5, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=int32),
     9)



Our loss is the output


```python
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

model.compile(optimizer="adam", loss=lambda y_true, y_pred: y_pred)
early_stopping = EarlyStopping(patience=40,monitor='val_loss', restore_best_weights=True)
```

Preprocessed texts don't fit in ram, if we were using the library this would be done internally


```python
import numpy as np

def generator_for_model(texts, tags_prep, lens, batch_size=256):
    texts = texts.copy()
    while True:
        perm = np.random.permutation(len(texts))
        texts = np.asarray(texts)[perm].tolist()
        tags_prep = tags_prep[perm]
        lens = lens[perm]
        for i in range(0, len(texts), batch_size):
            texts_inp1 = word_emb(word_emb.preprocess_texts(texts[i:i+batch_size]))
            texts_inp2 = char_digest_per_word(char_digest_per_word.preprocess_texts(texts[i:i+batch_size]))
            yield [texts_inp1, texts_inp2, tags_prep[i:i+batch_size], lens[i:i+batch_size]], tags_prep[i:i+batch_size]
```


```python
train_generator = generator_for_model(train_texts, train_tags_prep, train_lens)
valid_generator = generator_for_model(valid_texts, valid_tags_prep, valid_lens)
```


```python
hst = model.fit(train_generator, epochs=400, steps_per_epoch=len(train_texts)//256 + 1,
                validation_data=valid_generator, validation_steps=len(valid_texts)//256 + 1,
                callbacks=[early_stopping])
```

    /tmp/ipykernel_4924/2906041889.py:7: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      texts = np.asarray(texts)[perm].tolist()


    Epoch 1/400
    59/59 [==============================] - 68s 943ms/step - loss: 5136.9131 - val_loss: 4754.5044
    Epoch 2/400
    59/59 [==============================] - 48s 809ms/step - loss: 4125.1953 - val_loss: 4173.6631
    Epoch 3/400
    59/59 [==============================] - 48s 808ms/step - loss: 3723.1543 - val_loss: 3824.8733
    Epoch 4/400
    59/59 [==============================] - 48s 821ms/step - loss: 3483.1428 - val_loss: 3652.2227
    Epoch 5/400
    59/59 [==============================] - 48s 820ms/step - loss: 3305.0483 - val_loss: 3418.8108
    Epoch 6/400
    59/59 [==============================] - 48s 822ms/step - loss: 3155.9189 - val_loss: 3356.5911
    Epoch 7/400
    59/59 [==============================] - 50s 841ms/step - loss: 3030.2600 - val_loss: 3164.5894
    Epoch 8/400
    59/59 [==============================] - 49s 825ms/step - loss: 2923.7966 - val_loss: 3101.0005
    Epoch 9/400
    59/59 [==============================] - 48s 814ms/step - loss: 2832.1487 - val_loss: 3000.5598
    Epoch 10/400
    59/59 [==============================] - 49s 829ms/step - loss: 2751.9807 - val_loss: 2897.0979
    Epoch 11/400
    59/59 [==============================] - 48s 814ms/step - loss: 2686.6812 - val_loss: 2851.6726
    Epoch 12/400
    59/59 [==============================] - 47s 795ms/step - loss: 2628.6565 - val_loss: 2814.3750
    Epoch 13/400
    59/59 [==============================] - 47s 792ms/step - loss: 2582.7039 - val_loss: 2732.2253
    Epoch 14/400
    59/59 [==============================] - 46s 789ms/step - loss: 2541.5037 - val_loss: 2702.0347
    Epoch 15/400
    59/59 [==============================] - 46s 789ms/step - loss: 2504.1814 - val_loss: 2670.0986
    Epoch 16/400
    59/59 [==============================] - 47s 796ms/step - loss: 2473.8137 - val_loss: 2649.4033
    Epoch 17/400
    59/59 [==============================] - 47s 791ms/step - loss: 2445.5342 - val_loss: 2590.4839
    Epoch 18/400
    59/59 [==============================] - 47s 798ms/step - loss: 2418.7954 - val_loss: 2596.5842
    Epoch 19/400
    59/59 [==============================] - 48s 817ms/step - loss: 2397.3411 - val_loss: 2566.2034
    Epoch 20/400
    59/59 [==============================] - 47s 799ms/step - loss: 2376.1152 - val_loss: 2488.3650
    Epoch 21/400
    59/59 [==============================] - 47s 802ms/step - loss: 2354.0979 - val_loss: 2510.8647
    Epoch 22/400
    59/59 [==============================] - 46s 782ms/step - loss: 2304.4912 - val_loss: 2436.1367
    Epoch 23/400
    59/59 [==============================] - 46s 783ms/step - loss: 2270.8281 - val_loss: 2471.4529
    Epoch 24/400
    59/59 [==============================] - 47s 790ms/step - loss: 2247.2590 - val_loss: 2373.6514
    Epoch 25/400
    59/59 [==============================] - 47s 802ms/step - loss: 2211.4924 - val_loss: 2382.1360
    Epoch 26/400
    59/59 [==============================] - 47s 798ms/step - loss: 2113.9595 - val_loss: 2156.2512
    Epoch 27/400
    59/59 [==============================] - 46s 783ms/step - loss: 1829.9944 - val_loss: 1681.5222
    Epoch 28/400
    59/59 [==============================] - 46s 788ms/step - loss: 1508.4958 - val_loss: 1532.1682
    Epoch 29/400
    59/59 [==============================] - 46s 782ms/step - loss: 1392.7432 - val_loss: 1428.9027
    Epoch 30/400
    59/59 [==============================] - 47s 793ms/step - loss: 1306.3041 - val_loss: 1361.3661
    Epoch 31/400
    59/59 [==============================] - 46s 789ms/step - loss: 1241.7709 - val_loss: 1307.8262
    Epoch 32/400
    59/59 [==============================] - 46s 774ms/step - loss: 1185.1738 - val_loss: 1235.7374
    Epoch 33/400
    59/59 [==============================] - 46s 781ms/step - loss: 1138.2565 - val_loss: 1215.7477
    Epoch 34/400
    59/59 [==============================] - 47s 801ms/step - loss: 1099.7321 - val_loss: 1162.0999
    Epoch 35/400
    59/59 [==============================] - 46s 774ms/step - loss: 1065.2351 - val_loss: 1165.6729
    Epoch 36/400
    59/59 [==============================] - 48s 813ms/step - loss: 1038.6534 - val_loss: 1096.5824
    Epoch 37/400
    59/59 [==============================] - 46s 788ms/step - loss: 1012.3759 - val_loss: 1096.6584
    Epoch 38/400
    59/59 [==============================] - 48s 809ms/step - loss: 987.4750 - val_loss: 1064.0221
    Epoch 39/400
    59/59 [==============================] - 48s 820ms/step - loss: 969.8301 - val_loss: 1049.3406
    Epoch 40/400
    59/59 [==============================] - 46s 785ms/step - loss: 950.4096 - val_loss: 1044.1338
    Epoch 41/400
    59/59 [==============================] - 47s 800ms/step - loss: 934.3626 - val_loss: 1030.5472
    Epoch 42/400
    59/59 [==============================] - 46s 785ms/step - loss: 914.9908 - val_loss: 1013.6733
    Epoch 43/400
    59/59 [==============================] - 46s 788ms/step - loss: 899.6580 - val_loss: 1008.8781
    Epoch 44/400
    59/59 [==============================] - 46s 785ms/step - loss: 886.6391 - val_loss: 995.3293
    Epoch 45/400
    59/59 [==============================] - 48s 818ms/step - loss: 875.8398 - val_loss: 997.0549
    Epoch 46/400
    59/59 [==============================] - 47s 801ms/step - loss: 863.2900 - val_loss: 984.3563
    Epoch 47/400
    59/59 [==============================] - 47s 802ms/step - loss: 857.0876 - val_loss: 991.5219
    Epoch 48/400
    59/59 [==============================] - 49s 825ms/step - loss: 845.9821 - val_loss: 944.6074
    Epoch 49/400
    59/59 [==============================] - 46s 782ms/step - loss: 838.5079 - val_loss: 948.9692
    Epoch 50/400
    59/59 [==============================] - 47s 803ms/step - loss: 834.2989 - val_loss: 967.4363
    Epoch 51/400
    59/59 [==============================] - 47s 795ms/step - loss: 823.6083 - val_loss: 957.6635
    Epoch 52/400
    59/59 [==============================] - 46s 789ms/step - loss: 819.3171 - val_loss: 933.3299
    Epoch 53/400
    59/59 [==============================] - 47s 796ms/step - loss: 810.8563 - val_loss: 961.1999
    Epoch 54/400
    59/59 [==============================] - 47s 802ms/step - loss: 803.9817 - val_loss: 932.4126
    Epoch 55/400
    59/59 [==============================] - 47s 800ms/step - loss: 798.3369 - val_loss: 935.8329
    Epoch 56/400
    59/59 [==============================] - 46s 787ms/step - loss: 793.5576 - val_loss: 933.3190
    Epoch 57/400
    59/59 [==============================] - 46s 784ms/step - loss: 786.1114 - val_loss: 930.7341
    Epoch 58/400
    59/59 [==============================] - 46s 785ms/step - loss: 781.6781 - val_loss: 925.9508
    Epoch 59/400
    59/59 [==============================] - 46s 782ms/step - loss: 778.5130 - val_loss: 916.4297
    Epoch 60/400
    59/59 [==============================] - 46s 788ms/step - loss: 772.3932 - val_loss: 930.2745
    Epoch 61/400
    59/59 [==============================] - 46s 786ms/step - loss: 769.5518 - val_loss: 899.9677
    Epoch 62/400
    59/59 [==============================] - 46s 784ms/step - loss: 765.5557 - val_loss: 902.3265
    Epoch 63/400
    59/59 [==============================] - 46s 786ms/step - loss: 760.4366 - val_loss: 934.2704
    Epoch 64/400
    59/59 [==============================] - 46s 786ms/step - loss: 756.3931 - val_loss: 883.0938
    Epoch 65/400
    59/59 [==============================] - 46s 782ms/step - loss: 751.1208 - val_loss: 901.1564
    Epoch 66/400
    59/59 [==============================] - 46s 782ms/step - loss: 745.7216 - val_loss: 898.8864
    Epoch 67/400
    59/59 [==============================] - 46s 786ms/step - loss: 742.5194 - val_loss: 900.4983
    Epoch 68/400
    59/59 [==============================] - 47s 795ms/step - loss: 740.2647 - val_loss: 867.6713
    Epoch 69/400
    59/59 [==============================] - 46s 788ms/step - loss: 738.0363 - val_loss: 896.4472
    Epoch 70/400
    59/59 [==============================] - 46s 782ms/step - loss: 735.3856 - val_loss: 886.0616
    Epoch 71/400
    59/59 [==============================] - 46s 785ms/step - loss: 730.4325 - val_loss: 885.0441
    Epoch 72/400
    59/59 [==============================] - 46s 783ms/step - loss: 725.7354 - val_loss: 872.2396
    Epoch 73/400
    59/59 [==============================] - 46s 782ms/step - loss: 724.6079 - val_loss: 886.7775
    Epoch 74/400
    59/59 [==============================] - 47s 790ms/step - loss: 720.7198 - val_loss: 853.1366
    Epoch 75/400
    59/59 [==============================] - 46s 784ms/step - loss: 719.5021 - val_loss: 888.6414
    Epoch 76/400
    59/59 [==============================] - 47s 794ms/step - loss: 719.9547 - val_loss: 853.9789
    Epoch 77/400
    59/59 [==============================] - 46s 783ms/step - loss: 713.3592 - val_loss: 892.4611
    Epoch 78/400
    59/59 [==============================] - 46s 788ms/step - loss: 710.7239 - val_loss: 863.5499
    Epoch 79/400
    59/59 [==============================] - 47s 792ms/step - loss: 708.9562 - val_loss: 870.0413
    Epoch 80/400
    59/59 [==============================] - 46s 786ms/step - loss: 705.3092 - val_loss: 878.5925
    Epoch 81/400
    59/59 [==============================] - 46s 786ms/step - loss: 705.2905 - val_loss: 868.2865
    Epoch 82/400
    59/59 [==============================] - 46s 783ms/step - loss: 702.9382 - val_loss: 860.3056
    Epoch 83/400
    59/59 [==============================] - 46s 778ms/step - loss: 708.2054 - val_loss: 877.0040
    Epoch 84/400
    59/59 [==============================] - 46s 781ms/step - loss: 701.7367 - val_loss: 860.1691
    Epoch 85/400
    59/59 [==============================] - 46s 786ms/step - loss: 698.2739 - val_loss: 857.2281
    Epoch 86/400
    59/59 [==============================] - 46s 783ms/step - loss: 694.3503 - val_loss: 844.1545
    Epoch 87/400
    59/59 [==============================] - 46s 783ms/step - loss: 691.8611 - val_loss: 861.2796
    Epoch 88/400
    59/59 [==============================] - 46s 776ms/step - loss: 691.5272 - val_loss: 852.3960
    Epoch 89/400
    59/59 [==============================] - 46s 780ms/step - loss: 691.0486 - val_loss: 845.2658
    Epoch 90/400
    59/59 [==============================] - 46s 785ms/step - loss: 687.0771 - val_loss: 853.6535
    Epoch 91/400
    59/59 [==============================] - 46s 781ms/step - loss: 684.7747 - val_loss: 834.5568
    Epoch 92/400
    59/59 [==============================] - 46s 782ms/step - loss: 683.1312 - val_loss: 857.4968
    Epoch 93/400
    59/59 [==============================] - 46s 783ms/step - loss: 682.8558 - val_loss: 854.6352
    Epoch 94/400
    59/59 [==============================] - 49s 832ms/step - loss: 681.1857 - val_loss: 830.2047
    Epoch 95/400
    59/59 [==============================] - 47s 791ms/step - loss: 681.3492 - val_loss: 857.1024
    Epoch 96/400
    59/59 [==============================] - 46s 785ms/step - loss: 680.1462 - val_loss: 848.6035
    Epoch 97/400
    59/59 [==============================] - 46s 785ms/step - loss: 678.7196 - val_loss: 841.7513
    Epoch 98/400
    59/59 [==============================] - 46s 786ms/step - loss: 678.0618 - val_loss: 840.0606
    Epoch 99/400
    59/59 [==============================] - 46s 785ms/step - loss: 675.6106 - val_loss: 836.2761
    Epoch 100/400
    59/59 [==============================] - 46s 787ms/step - loss: 676.5831 - val_loss: 848.8679
    Epoch 101/400
    59/59 [==============================] - 46s 785ms/step - loss: 676.2980 - val_loss: 835.7365
    Epoch 102/400
    59/59 [==============================] - 46s 783ms/step - loss: 674.6821 - val_loss: 845.1892
    Epoch 103/400
    59/59 [==============================] - 46s 786ms/step - loss: 670.7278 - val_loss: 843.0674
    Epoch 104/400
    59/59 [==============================] - 46s 786ms/step - loss: 670.7476 - val_loss: 826.9028
    Epoch 105/400
    59/59 [==============================] - 46s 782ms/step - loss: 668.1880 - val_loss: 852.6885
    Epoch 106/400
    59/59 [==============================] - 46s 782ms/step - loss: 666.8660 - val_loss: 814.8812
    Epoch 107/400
    59/59 [==============================] - 46s 782ms/step - loss: 667.2969 - val_loss: 847.1368
    Epoch 108/400
    59/59 [==============================] - 47s 791ms/step - loss: 666.3581 - val_loss: 835.0107
    Epoch 109/400
    59/59 [==============================] - 46s 788ms/step - loss: 666.0004 - val_loss: 832.0311
    Epoch 110/400
    59/59 [==============================] - 46s 784ms/step - loss: 668.5718 - val_loss: 838.2491
    Epoch 111/400
    59/59 [==============================] - 46s 786ms/step - loss: 664.4260 - val_loss: 830.7400
    Epoch 112/400
    59/59 [==============================] - 46s 777ms/step - loss: 663.3076 - val_loss: 826.9965
    Epoch 113/400
    59/59 [==============================] - 46s 786ms/step - loss: 662.3937 - val_loss: 822.2523
    Epoch 114/400
    59/59 [==============================] - 47s 791ms/step - loss: 662.5190 - val_loss: 832.6190
    Epoch 115/400
    59/59 [==============================] - 46s 783ms/step - loss: 658.7273 - val_loss: 834.6021
    Epoch 116/400
    59/59 [==============================] - 46s 789ms/step - loss: 658.5717 - val_loss: 837.6925
    Epoch 117/400
    59/59 [==============================] - 46s 788ms/step - loss: 656.6953 - val_loss: 804.3206
    Epoch 118/400
    59/59 [==============================] - 46s 789ms/step - loss: 654.8416 - val_loss: 835.0590
    Epoch 119/400
    59/59 [==============================] - 46s 782ms/step - loss: 652.7203 - val_loss: 811.7708
    Epoch 120/400
    59/59 [==============================] - 46s 785ms/step - loss: 653.1294 - val_loss: 822.7291
    Epoch 121/400
    59/59 [==============================] - 47s 790ms/step - loss: 649.9742 - val_loss: 811.7941
    Epoch 122/400
    59/59 [==============================] - 46s 790ms/step - loss: 647.6527 - val_loss: 820.6454
    Epoch 123/400
    59/59 [==============================] - 47s 791ms/step - loss: 643.0724 - val_loss: 824.5834
    Epoch 124/400
    59/59 [==============================] - 46s 789ms/step - loss: 642.9667 - val_loss: 812.6609
    Epoch 125/400
    59/59 [==============================] - 46s 786ms/step - loss: 639.7587 - val_loss: 819.6337
    Epoch 126/400
    59/59 [==============================] - 46s 781ms/step - loss: 637.1480 - val_loss: 810.6826
    Epoch 127/400
    59/59 [==============================] - 46s 788ms/step - loss: 635.5370 - val_loss: 802.2789
    Epoch 128/400
    59/59 [==============================] - 46s 786ms/step - loss: 634.4254 - val_loss: 822.8185
    Epoch 129/400
    59/59 [==============================] - 46s 785ms/step - loss: 635.3126 - val_loss: 807.8918
    Epoch 130/400
    59/59 [==============================] - 46s 781ms/step - loss: 633.9419 - val_loss: 806.1224
    Epoch 131/400
    59/59 [==============================] - 46s 785ms/step - loss: 631.6010 - val_loss: 787.3816
    Epoch 132/400
    59/59 [==============================] - 46s 783ms/step - loss: 629.2901 - val_loss: 805.8433
    Epoch 133/400
    59/59 [==============================] - 46s 788ms/step - loss: 629.0173 - val_loss: 769.3822
    Epoch 134/400
    59/59 [==============================] - 46s 781ms/step - loss: 627.8366 - val_loss: 821.0234
    Epoch 135/400
    59/59 [==============================] - 46s 784ms/step - loss: 625.2507 - val_loss: 797.5182
    Epoch 136/400
    59/59 [==============================] - 47s 792ms/step - loss: 623.7414 - val_loss: 782.8862
    Epoch 137/400
    59/59 [==============================] - 46s 782ms/step - loss: 622.3116 - val_loss: 800.6243
    Epoch 138/400
    59/59 [==============================] - 46s 785ms/step - loss: 623.4420 - val_loss: 791.8024
    Epoch 139/400
    59/59 [==============================] - 46s 785ms/step - loss: 620.5997 - val_loss: 797.0449
    Epoch 140/400
    59/59 [==============================] - 46s 784ms/step - loss: 623.3973 - val_loss: 796.7684
    Epoch 141/400
    59/59 [==============================] - 46s 789ms/step - loss: 622.0497 - val_loss: 791.5201
    Epoch 142/400
    59/59 [==============================] - 46s 785ms/step - loss: 622.4817 - val_loss: 802.7406
    Epoch 143/400
    59/59 [==============================] - 46s 780ms/step - loss: 617.7530 - val_loss: 776.7987
    Epoch 144/400
    59/59 [==============================] - 46s 784ms/step - loss: 617.7052 - val_loss: 796.7974
    Epoch 145/400
    59/59 [==============================] - 46s 783ms/step - loss: 619.8998 - val_loss: 784.9045
    Epoch 146/400
    59/59 [==============================] - 46s 780ms/step - loss: 616.9827 - val_loss: 801.0306
    Epoch 147/400
    59/59 [==============================] - 46s 785ms/step - loss: 617.4465 - val_loss: 782.8989
    Epoch 148/400
    59/59 [==============================] - 47s 805ms/step - loss: 616.4445 - val_loss: 790.6892
    Epoch 149/400
    59/59 [==============================] - 47s 797ms/step - loss: 616.8898 - val_loss: 789.3041
    Epoch 150/400
    59/59 [==============================] - 46s 785ms/step - loss: 614.3298 - val_loss: 789.2144
    Epoch 151/400
    59/59 [==============================] - 46s 785ms/step - loss: 614.5011 - val_loss: 799.8588
    Epoch 152/400
    59/59 [==============================] - 46s 780ms/step - loss: 612.8457 - val_loss: 798.0997
    Epoch 153/400
    59/59 [==============================] - 46s 785ms/step - loss: 612.5789 - val_loss: 778.6464
    Epoch 154/400
    59/59 [==============================] - 47s 805ms/step - loss: 610.8384 - val_loss: 782.2866
    Epoch 155/400
    59/59 [==============================] - 46s 781ms/step - loss: 609.9700 - val_loss: 789.5822
    Epoch 156/400
    59/59 [==============================] - 46s 781ms/step - loss: 612.5296 - val_loss: 782.2175
    Epoch 157/400
    59/59 [==============================] - 46s 783ms/step - loss: 609.8220 - val_loss: 792.0643
    Epoch 158/400
    59/59 [==============================] - 46s 781ms/step - loss: 608.2350 - val_loss: 774.8680
    Epoch 159/400
    59/59 [==============================] - 46s 789ms/step - loss: 610.0908 - val_loss: 786.3444
    Epoch 160/400
    59/59 [==============================] - 46s 788ms/step - loss: 608.8103 - val_loss: 788.5714
    Epoch 161/400
    59/59 [==============================] - 46s 786ms/step - loss: 607.1669 - val_loss: 786.0401
    Epoch 162/400
    59/59 [==============================] - 46s 788ms/step - loss: 606.4327 - val_loss: 782.6342
    Epoch 163/400
    59/59 [==============================] - 47s 791ms/step - loss: 604.2177 - val_loss: 789.0533
    Epoch 164/400
    59/59 [==============================] - 46s 786ms/step - loss: 601.6615 - val_loss: 800.0118
    Epoch 165/400
    59/59 [==============================] - 46s 785ms/step - loss: 602.8777 - val_loss: 787.3075
    Epoch 166/400
    59/59 [==============================] - 46s 786ms/step - loss: 602.3177 - val_loss: 770.3492
    Epoch 167/400
    59/59 [==============================] - 46s 783ms/step - loss: 605.8418 - val_loss: 776.7177
    Epoch 168/400
    59/59 [==============================] - 46s 779ms/step - loss: 603.6479 - val_loss: 780.5766
    Epoch 169/400
    59/59 [==============================] - 47s 792ms/step - loss: 602.8694 - val_loss: 795.9206
    Epoch 170/400
    59/59 [==============================] - 46s 788ms/step - loss: 603.5446 - val_loss: 778.7560
    Epoch 171/400
    59/59 [==============================] - 46s 782ms/step - loss: 601.8182 - val_loss: 773.0528
    Epoch 172/400
    59/59 [==============================] - 46s 789ms/step - loss: 601.2823 - val_loss: 799.2163
    Epoch 173/400
    59/59 [==============================] - 46s 784ms/step - loss: 600.5944 - val_loss: 777.5840


## Evaluation


```python
from tensorflow_addons.text import crf_decode
```


```python
transition_potentials = model.layers[-1].get_weights()[0]
```


```python
!pip install matplotlib
```

    Collecting matplotlib
      Downloading matplotlib-3.5.1-cp39-cp39-manylinux_2_5_x86_64.manylinux1_x86_64.whl (11.2 MB)
    [K     |████████████████████████████████| 11.2 MB 4.3 MB/s eta 0:00:01
    [?25hRequirement already satisfied: pyparsing>=2.2.1 in /home/jian01/.pyenv/versions/3.9.5/envs/meli/lib/python3.9/site-packages (from matplotlib) (3.0.7)
    Collecting fonttools>=4.22.0
      Downloading fonttools-4.29.1-py3-none-any.whl (895 kB)
    [K     |████████████████████████████████| 895 kB 22.9 MB/s eta 0:00:01
    [?25hCollecting cycler>=0.10
      Downloading cycler-0.11.0-py3-none-any.whl (6.4 kB)
    Requirement already satisfied: numpy>=1.17 in /home/jian01/.pyenv/versions/3.9.5/envs/meli/lib/python3.9/site-packages (from matplotlib) (1.22.1)
    Collecting kiwisolver>=1.0.1
      Downloading kiwisolver-1.3.2-cp39-cp39-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.6 MB)
    [K     |████████████████████████████████| 1.6 MB 13.8 MB/s eta 0:00:01
    [?25hCollecting pillow>=6.2.0
      Downloading Pillow-9.0.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.3 MB)
    [K     |████████████████████████████████| 4.3 MB 7.8 MB/s eta 0:00:01
    [?25hRequirement already satisfied: packaging>=20.0 in /home/jian01/.pyenv/versions/3.9.5/envs/meli/lib/python3.9/site-packages (from matplotlib) (21.3)
    Requirement already satisfied: python-dateutil>=2.7 in /home/jian01/.pyenv/versions/3.9.5/envs/meli/lib/python3.9/site-packages (from matplotlib) (2.8.2)
    Requirement already satisfied: six>=1.5 in /home/jian01/.pyenv/versions/3.9.5/envs/meli/lib/python3.9/site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)
    Installing collected packages: pillow, kiwisolver, fonttools, cycler, matplotlib
    Successfully installed cycler-0.11.0 fonttools-4.29.1 kiwisolver-1.3.2 matplotlib-3.5.1 pillow-9.0.1
    [33mWARNING: You are using pip version 21.1.1; however, version 22.0.3 is available.
    You should consider upgrading via the '/home/jian01/.pyenv/versions/3.9.5/envs/meli/bin/python3.9 -m pip install --upgrade pip' command.[0m


We can see the transition potentials make sense


```python
import matplotlib.pyplot as plt
transitions = transition_potentials/np.mean(transition_potentials, axis=1)
new_indexes = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC']
transitions = transitions[[possible_tags.index(l) for l in new_indexes],:]
transitions = transitions[:,[possible_tags.index(l) for l in new_indexes]]
plt.imshow(transitions, cmap='Wistia')
plt.xticks(list(range(9)), new_indexes, rotation=90)
plt.yticks(list(range(9)), new_indexes)
plt.title("Transition potentials")
plt.show()
```


    
![png](_static/output_48_0.png)
    



```python
inference_model = models.Model(inputs=[inp1, inp2],
                               outputs=model.get_layer('logits').output)
```


```python
BATCH = 256
preds = []
for i in range(0, len(test_texts), BATCH):
    batch_texts = test_texts[i:i+BATCH]
    texts_inp1 = word_emb(word_emb.preprocess_texts(batch_texts))
    texts_inp2 = char_digest_per_word(char_digest_per_word.preprocess_texts(batch_texts))
    logits = inference_model.predict([texts_inp1, texts_inp2])
    temp_preds, _ = crf_decode(logits, transition_potentials, [113]*len(batch_texts))
    temp_preds = temp_preds.numpy().tolist()
    for i in range(len(temp_preds)):
        temp_preds[i] = temp_preds[i][:len(batch_texts[i])]
    preds += temp_preds
```


```python
test_texts[0]
```




    ['SOCCER',
     '-',
     'JAPAN',
     'GET',
     'LUCKY',
     'WIN',
     ',',
     'CHINA',
     'IN',
     'SURPRISE',
     'DEFEAT',
     '.']




```python
test_tags[0]
```




    ['O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'B-PER', 'O', 'O', 'O', 'O']




```python
[possible_tags[idx] for idx in preds[0]]
```




    ['O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O']




```python
!pip install sklearn
```

    Requirement already satisfied: sklearn in /home/jian01/.pyenv/versions/3.9.5/envs/meli/lib/python3.9/site-packages (0.0)
    Requirement already satisfied: scikit-learn in /home/jian01/.pyenv/versions/3.9.5/envs/meli/lib/python3.9/site-packages (from sklearn) (1.0.2)
    Requirement already satisfied: scipy>=1.1.0 in /home/jian01/.pyenv/versions/3.9.5/envs/meli/lib/python3.9/site-packages (from scikit-learn->sklearn) (1.7.3)
    Requirement already satisfied: joblib>=0.11 in /home/jian01/.pyenv/versions/3.9.5/envs/meli/lib/python3.9/site-packages (from scikit-learn->sklearn) (1.1.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /home/jian01/.pyenv/versions/3.9.5/envs/meli/lib/python3.9/site-packages (from scikit-learn->sklearn) (3.1.0)
    Requirement already satisfied: numpy>=1.14.6 in /home/jian01/.pyenv/versions/3.9.5/envs/meli/lib/python3.9/site-packages (from scikit-learn->sklearn) (1.22.1)
    [33mWARNING: You are using pip version 21.1.1; however, version 22.0.3 is available.
    You should consider upgrading via the '/home/jian01/.pyenv/versions/3.9.5/envs/meli/bin/python3.9 -m pip install --upgrade pip' command.[0m



```python
from sklearn.metrics import classification_report
```


```python
print( classification_report([t for ts in test_tags for t in ts[:113]], [possible_tags[idx] for idxs in preds for idx in idxs]) )
```

    /home/jian01/.pyenv/versions/3.9.5/envs/meli/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /home/jian01/.pyenv/versions/3.9.5/envs/meli/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))


                  precision    recall  f1-score   support
    
                       0.00      0.00      0.00         1
           B-LOC       0.88      0.80      0.84      1667
          B-MISC       0.73      0.24      0.36       702
           B-ORG       0.71      0.56      0.63      1661
           B-PER       0.95      0.70      0.81      1616
           I-LOC       0.63      0.40      0.49       256
          I-MISC       0.55      0.55      0.55       216
           I-ORG       0.57      0.75      0.65       835
           I-PER       0.95      0.92      0.93      1155
               O       0.96      0.99      0.97     38546
    
        accuracy                           0.93     46655
       macro avg       0.69      0.59      0.62     46655
    weighted avg       0.93      0.93      0.93     46655
    


    /home/jian01/.pyenv/versions/3.9.5/envs/meli/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))


## Serialization

We need to save the keras model, word emb and char digest by word.


```python
inference_model.save('...')
data1 = word_emb.serialize()
data2 = char_digest_per_word.serialize()
```
