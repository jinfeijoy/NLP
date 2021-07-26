#!/usr/bin/env python
# coding: utf-8

# In[36]:


import sys
import os
sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_nlp_pkg')
sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_model_explain_pkg')
import nlpbasic.textClean as textClean
import nlpbasic.docVectors as DocVector
import nlpbasic.dataExploration as DataExploration
import nlpbasic.lda as lda
import nlpbasic.tfidf as tfidf
import nlpbasic.text_summarize as txtsmr
import nlpbasic.word_embedding as wdembd

import model_explain.plot as meplot
import model_explain.shap as meshap

import data_visualization.distribution_plot as dbplot
from numpy import array,asarray,zeros
import numpy as np
import pandas as pd
import json
import re
import tensorflow as tf
import random
import spacy
nlp = spacy.load('en_core_web_sm')

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Embedding

datapath = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data'
datapath2 = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data\\ATIS'
#dataset: https://www.kaggle.com/hassanamin/atis-airlinetravelinformationsystem?select=atis_intents_train.csv
#dataset2: https://www.kaggle.com/elvinagammed/chatbots-intent-recognition-dataset/code


# ## Load Dataset

# **Chatboat**: https://www.kaggle.com/elvinagammed/chatbots-intent-recognition-dataset/code

# In[24]:


def preprocessing(line):
    line = re.sub(r'[^a-zA-z.?!\']', ' ', line)
    line = re.sub(r'[ ]+', ' ', line)
    return line
with open(os.path.join(datapath, 'Intent.json')) as f:
          intents = json.load(f)

# get text and intent title from json data: output is a dictionary
inputs, targets = [], []
classes = []
intent_doc = {}

for intent in intents['intents']:
    if intent['intent'] not in classes:
        classes.append(intent['intent'])
    if intent['intent'] not in intent_doc:
        intent_doc[intent['intent']] = []
        
    for text in intent['text']:
        inputs.append(preprocessing(text))
        targets.append(intent['intent'])
        
    for response in intent['responses']:
        intent_doc[intent['intent']].append(response)
        
#generate dataset
data = intents['intents']
dataset = pd.DataFrame(columns=['intent', 'text', 'response'])
for i in data:
    intent = i['intent']
    for t, r in zip(i['text'], i['responses']):
        row = {'intent': intent, 'text': t, 'response':r}
        dataset = dataset.append(row, ignore_index=True)


# In[42]:


len(dataset.intent.unique())


# In[31]:


X = [x for x in dataset.text]
y = pd.get_dummies(dataset.intent).values


# In[34]:


# load the whole embedding into memory
embeddings_index = dict()
embedding_dim = 100 
# download glove word embedding first and then load it with the following code
f = open('C:/ProgramData/Anaconda3/append_file/glove/glove.6B.100d.txt', encoding = 'utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs
f.close
print('loaded %s word vectors.' % len(embeddings_index))


# In[39]:


max_length = int(np.percentile(dataset.text.apply(lambda x: len(x.split())), 95))
# we also tried max length, but it cause overfitting

t = Tokenizer()
t.fit_on_texts(X)
# print("words with freq:", t.word_docs)

vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(X)
print('Encoding:\n', encoded_docs[0])
print('\nWord Indices:\n', [(t.index_word[i], i) for i in encoded_docs[0]])
print('vocab size:', vocab_size)
train_padded_docs = pad_sequences(encoded_docs, maxlen = max_length, padding = 'post')


# In[40]:


embedding_matrix = zeros((vocab_size, embedding_dim))
for word, idx_word in t.word_index.items():
    word_vector = embeddings_index.get(word)
    if word_vector is not None:
        embedding_matrix[idx_word] = word_vector


# In[43]:


model = Sequential(
    [
        Embedding(vocab_size, embedding_dim, weights = [embedding_matrix], input_length = max_length, trainable = False),
        Flatten(),
        Dense(embedding_dim, activation="relu", name="layer1"),
        Dense(22, activation = 'softmax', name="layer2")
        
    ]
)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
print(model.summary())


# In[44]:


history = model.fit(train_padded_docs, y, epochs = 10, verbose = 1, batch_size = 32)


# In[60]:


label_list = pd.get_dummies(dataset.intent).columns.tolist()
encoded_val_doc = t.texts_to_sequences(['You'])
padded_val_doc = pad_sequences(encoded_val_doc, maxlen = max_length, padding = 'post')
label_list[np.argmax(model.predict(padded_val_doc))]


# In[66]:


label_list = pd.get_dummies(dataset.intent).columns.tolist()
def response(sentence, maxlen):
    encoded_val_doc = t.texts_to_sequences([sentence])
    padded_val_doc = pad_sequences(encoded_val_doc, maxlen = maxlen, padding = 'post')

    # predict the category of input sentences
    pred_class = label_list[np.argmax(model.predict(padded_val_doc))]
    
    # choice a random response for predicted sentence
    return random.choice(intent_doc[pred_class]), pred_class

# chat with bot
print("Note: Enter 'quit' to break the loop.")
while True:
    input_ = input('You: ')
    if input_.lower() == 'quit':
        break
    res, typ = response(input_, max_length)
    print('Bot: {} -- TYPE: {}'.format(res, typ))
    print()


# **Introducing ATIS : Intent Classification Dataset**: https://www.kaggle.com/hassanamin/atis-airlinetravelinformationsystem

# In[20]:


atis_train = pd.read_csv(os.path.join(datapath2, 'atis_intents_train.csv'),header=None).rename(columns = {0: 'intent', 1: 'text'})


# In[21]:


atis_train.head(3)

