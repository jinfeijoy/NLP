#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_nlp_pkg')
sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_model_explain_pkg')
import nlpbasic.textClean as textClean
import nlpbasic.docVectors as DocVector
import nlpbasic.dataExploration as DataExploration
import nlpbasic.lda as lda
import nlpbasic.tfidf as tfidf

import model_explain.plot as meplot
import model_explain.shap as meshap

from numpy import array,asarray,zeros
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Flatten,Embedding
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import re
pd.set_option('display.max_colwidth', None)
root_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data'


# # Load Data and Basic Exploration

# In[2]:


djia_news = pd.read_csv(os.path.join(root_path, "djia_news.csv"))
nasdaq = pd.read_csv(os.path.join(root_path, "nasdaq.csv"))
class_names = ['negative', 'positive','neutral'] #0:neg 1:pos 2:neu


# In[3]:


djia_news.head(3)


# In[4]:


nasdaq.head(3)


# In[10]:


print('djia_news label',djia_news.Label.value_counts())
print('djia_news data', len(djia_news))
print('djia_news ticker', len(djia_news.Ticker.unique()))
print('----------------')
print('djia_news sample')
print(djia_news.Headline[0])


# In[11]:


print('nasdaq label',nasdaq.Label.value_counts())
print('nasdaq data', len(nasdaq))
print('nasdaq ticker', len(nasdaq.Ticker.unique()))
print('----------------')
print('nasdaq sample')
print(nasdaq.Headline[0])


# # NLP Exploration

# In[16]:


djia_tokens = textClean.pipeline(djia_news['Headline'].to_list(), multi_gram = [1], lower_case=True, deacc=False, 
                                 encoding='utf8', errors='strict', stem_lemma = 'lemma', tag_drop = ['V'], nltk_stop=True, 
                                 stop_word_list=[], remove_pattern = [], check_numbers=True, word_length=2, 
                                 remove_consecutives=True)


# In[26]:


nasdaq_tokens = textClean.pipeline(nasdaq['Headline'].to_list(), multi_gram = [1], lower_case=True, deacc=False, 
                                 encoding='utf8', errors='strict', stem_lemma = 'lemma', tag_drop = ['V'], nltk_stop=True, 
                                 stop_word_list=[], remove_pattern = ['http:','#', '@'], check_numbers=True, word_length=2, 
                                 remove_consecutives=True)


# In[23]:


top_10_freq_words = [i[0] for i in DataExploration.get_topn_freq_bow(djia_tokens, topn = 10)]
print('top 10 frequent words', top_10_freq_words)
top30tfidf = tfidf.get_top_n_tfidf_bow(djia_tokens, top_n_tokens = 30)
print('top 30 tfidf', top30tfidf)
no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(djia_tokens, num_topics = no_topics)
lda.lda_topics(lda_allbow)


# In[20]:


DataExploration.generate_word_cloud(djia_tokens)


# In[27]:


top_10_freq_words = [i[0] for i in DataExploration.get_topn_freq_bow(nasdaq_tokens, topn = 10)]
print('top 10 frequent words', top_10_freq_words)
top30tfidf = tfidf.get_top_n_tfidf_bow(nasdaq_tokens, top_n_tokens = 30)
print('top 30 tfidf', top30tfidf)
no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(nasdaq_tokens, num_topics = no_topics)
lda.lda_topics(lda_allbow)


# In[28]:


DataExploration.generate_word_cloud(nasdaq_tokens)


# # Word Embedding + RNN

# In[3]:


nasdaq['tokens'] = textClean.pipeline(nasdaq['Headline'].to_list(), multi_gram = [1], lower_case=True, deacc=False, 
                                 encoding='utf8', errors='strict', stem_lemma = 'lemma', tag_drop = ['V'], nltk_stop=True, 
                                 stop_word_list=[], remove_pattern = ['http:','#', '@'], check_numbers=True, word_length=2, 
                                 remove_consecutives=True)


# In[30]:


nasdaq.head(3)


# In[4]:


train_index, test_index= train_test_split(nasdaq.index , test_size = 0.33, random_state = 42)
X_train = nasdaq[nasdaq.index.isin(train_index)][['Headline']]
X_test = nasdaq[nasdaq.index.isin(test_index)][['Headline']]
y_train = pd.get_dummies(nasdaq[nasdaq.index.isin(train_index)]['Label']).values
y_test = pd.get_dummies(nasdaq[nasdaq.index.isin(test_index)]['Label']).values
X_train = [i for i in X_train.Headline]
X_test = [i for i in X_test.Headline]


# In[6]:


nasdaq['len'] = nasdaq.Headline.apply(lambda x: len(x))


# In[7]:


max_length = int(nasdaq.len.quantile(0.99))
max_length


# In[8]:


t = Tokenizer()
t.fit_on_texts(X_train)

vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(X_train)
padded_docs = pad_sequences(encoded_docs, maxlen = max_length, padding = 'post')
print('Encoding:\n', encoded_docs[0])
print('\nText:\n', list(X_train)[0])
print('\nWord Indices:\n', [(t.index_word[i], i) for i in encoded_docs[0]])
encoded_test_doc = t.texts_to_sequences(X_test)
padded_test_docs = pad_sequences(encoded_test_doc, maxlen = max_length, padding = 'post')

# load the whole embedding into memory
embeddings_index = dict()
# download glove word embedding first and then load it with the following code
f = open('C:/ProgramData/Anaconda3/append_file/glove/glove.6B.100d.txt', encoding = 'utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs
f.close
print('loaded %s word vectors.' % len(embeddings_index))

# Initialize a matrix with zeros having dimensions equivalent to vocab size and 100
embedding_matrix = zeros((vocab_size, 100))
for word, idx_word in t.word_index.items():
    word_vector = embeddings_index.get(word)
    if word_vector is not None:
        embedding_matrix[idx_word] = word_vector
print('word:', t.index_word[1])
print('Embedding:\n', embedding_matrix[1])
print('length of embedding matrix is:', len(embedding_matrix))
print('vocabulary size is %s.' % vocab_size)


# In[9]:


model = Sequential(
    [
        Embedding(vocab_size, 100, weights = [embedding_matrix], input_length = max_length, trainable = False),
        Flatten(),
        Dense(100, activation="relu", name="layer1"),
        Dense(3, activation = 'softmax', name="layer2")
        
    ]
)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
print(model.summary())


# In[12]:


# fit the model
history = model.fit(padded_docs, 
                    y_train, 
                    epochs = 10, 
                    verbose = 1, 
                    batch_size = 32, 
                    validation_data = (padded_test_docs, y_test)
                   )


# In[13]:


target_map = {'neg':0,'pos':1,'neu':2}
predsTest = pd.DataFrame(model.predict(padded_test_docs), columns = ['neg','pos','neu'])
predsTest['pred'] = predsTest.idxmax(axis = 1)
predsTest['pred_convert'] = predsTest['pred'].apply(lambda x: target_map[x])


# In[14]:


roundedPredsTest = predsTest.pred_convert
print('Confusion Matrix: Positive is class 1 and Negative is class 0')
cf_matrix = confusion_matrix(nasdaq[nasdaq.index.isin(test_index)]['Label'], roundedPredsTest, labels = [0,1,2])
print(cf_matrix)
meplot.cf_matrix_heatmap(cf_matrix, class_names)


# In[17]:


nasdaq[nasdaq.index.isin(test_index)]['Label'].value_counts()


# In[70]:


print(classification_report(nasdaq[nasdaq.index.isin(test_index)]['Label'],roundedPredsTest))

