#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import os
import tensorflow as tf


# In[2]:


import re
import json
from konlpy.tag import Okt
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tqdm import tqdm


# In[3]:


train = pd.read_table('Desktop/인공지능해커톤/train.txt', header = None)
train_data = train[1]
train_label = train[0]


# In[ ]:


def preprocessing(review, okt, remove_stopwords = False, stop_words = []):
    review_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", review)
    word_review = okt.morphs(review_text, stem=True)
    
    if remove_stopwords:
        word_review = [token for token in word_review if not token in stop_words]
    return word_review

stop_words = [ '은', '는', '이', '가', '하', '아', '것', '들','의', '있', '되', '수', '보', '주', '등', '한', '을', '를', '으로', '로']
okt = Okt()
clean_train_review = []

for review in tqdm(train_data):
    if type(review) == str:
        clean_train_review.append(preprocessing(review, okt, remove_stopwords = True, stop_words=stop_words))
    else:
        clean_train_review.append([])  

train_tokenizer = Tokenizer()
train_tokenizer.fit_on_texts(clean_train_review)
train_sequences = train_tokenizer.texts_to_sequences(clean_train_review)
MAX_SEQUENCE_LENGTH = 30 
train_inputs = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post') 


# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(train_label)
train_label = encoder.transform(train_label)
real_label = encoder.inverse_transform(train_label)

LABEL = [[0 for col in range(2)] for row in range(83712)]
for i in range(83712):
    LABEL[i][0] = train_label[i]
    LABEL[i][1] = real_label[i]
LABEL = list(set(map(tuple, LABEL)))


# In[ ]:


def vectorize_sequences(sequences, dimension=6872):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(train_inputs) 


# In[ ]:


import random
tmp = [[x,y] for x, y in zip(train_data, train_label)]
random.shuffle(tmp)


# In[ ]:


def to_one_hot(labels, dimension=785):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
one_hot_train_labels = to_one_hot(train_label)


# In[ ]:


from keras import models
from keras import layers
from keras.layers import Flatten, Dropout

x_val = x_train[:20000]
partial_x_train = x_train[20000:]

y_val = one_hot_train_labels[:20000]
partial_y_train = one_hot_train_labels[20000:]

model = models.Sequential()
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(785, activation='softmax'))

from keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), 
              metrics=['accuracy'])

model.fit(partial_x_train,
          partial_y_train,
          epochs=5,
          batch_size=512,
          validation_data=(x_val, y_val))

from keras.models import load_model
model.save('model1.h5')

