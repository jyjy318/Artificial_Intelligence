#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. 사용할 패키지 불러오기
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import re
from sklearn.preprocessing import LabelEncoder
from konlpy.tag import Okt
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tqdm import tqdm
from keras import models
from keras import layers
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

# 2. 테스트에 사용할 데이터 준비하기
test = pd.read_table('test.txt', header = None)
train = pd.read_table('train.txt', header = None)
train_data = train[1]
train_label = train[0]
test_data = test[1]
test_label = test[0]
stop_words = [ '은', '는', '이', '가', '하', '아', '것', '들','의', '있', '되', '수', '보', '주', '등', '한', '을', '를', '으로', '로']
okt = Okt()

def preprocessing(review, okt, remove_stopwords = False, stop_words = []):
    review_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", review)
    word_review = okt.morphs(review_text, stem=True)
    
    if remove_stopwords:
        word_review = [token for token in word_review if not token in stop_words]
    return word_review

def vectorize_sequences(sequences, dimension= 6872):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

##train data전처리
clean_train = []

for review in tqdm(train_data):
    if type(review) == str:
        clean_train.append(preprocessing(review, okt, remove_stopwords = True, stop_words=stop_words))
    else:
        clean_train.append([])  

train_tokenizer = Tokenizer()
train_tokenizer.fit_on_texts(clean_train)
train_sequences = train_tokenizer.texts_to_sequences(clean_train)

MAX_SEQUENCE_LENGTH = 30 
train_inputs = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post') 

encoder = LabelEncoder()
encoder.fit(train_label)
train_label = encoder.transform(train_label)
real_label = encoder.inverse_transform(train_label)

LABEL = [[0 for col in range(2)] for row in range(83712)]
for i in range(83712):
    LABEL[i][0] = train_label[i]
    LABEL[i][1] = real_label[i]
LABEL = list(set(map(tuple, LABEL)))

##test data전처리  
clean_test = []

for x in tqdm(test_data):
    if type(x) == str:
        clean_test.append(preprocessing(x, okt, remove_stopwords = True, stop_words=stop_words))
    else:
        clean_test.append([])

test_tokenizer = Tokenizer()
test_tokenizer.fit_on_texts(clean_train)
test_sequences = test_tokenizer.texts_to_sequences(clean_test)

MAX_SEQUENCE_LENGTH = 30 # 문장 최대 길이
test_inputs = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
x_test = vectorize_sequences(test_inputs)

# 3. 모델 불러오기
from keras.models import load_model
model = load_model('model.h5')

# 4. 모델 사용하기
predictions = model.predict(x_test)

test_predict = []
for i in range(len(test_data)):
    a = np.argmax(predictions[i])
    test_predict.append(a)

PREDICT = []
for i in range (len(test_data)):
    for j in range(len(LABEL)):
        if test_predict[i] == LABEL[j][0]:
            PREDICT.append(LABEL[j][1])
            
with open('result1.txt', 'w') as f:
    for line in PREDICT:
        f.write(line)
        f.write("\n")


# In[ ]:




