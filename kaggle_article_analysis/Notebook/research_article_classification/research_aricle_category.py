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
root_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data\\research_article_category'


# In[54]:


def generate_target(data):
    tmp = data.copy()
    tmp.set_index(['ID','TITLE','ABSTRACT'],inplace=True)
    tmp['LABEL'] = tmp.idxmax(axis=1)
    tmp = tmp.reset_index()
    tmp = tmp[['ID','TITLE','ABSTRACT','LABEL']]
    return tmp


# In[55]:


train = pd.read_csv(os.path.join(root_path, "train.csv"))
test = pd.read_csv(os.path.join(root_path, "test.csv"))
train = generate_target(train)
train.head(3)


# In[56]:


print(train['LABEL'].value_counts())


# In[ ]:





# In[38]:


train['title_tokens'] = textClean.pipeline(train['TITLE'].to_list(), multi_gram = [1], lower_case=True, deacc=False, 
                                     encoding='utf8', errors='strict', stem_lemma = 'lemma', tag_drop = [], nltk_stop=True, 
                                     stop_word_list=[], remove_pattern = [], check_numbers=True, word_length=2, 
                                     remove_consecutives=True)


# In[43]:


selected_tokens = train[train.LABEL == 'Mathematics']['title_tokens'].to_list()
top_10_freq_words = [i[0] for i in DataExploration.get_topn_freq_bow(selected_tokens, topn = 10)]
print('top 10 frequent words', top_10_freq_words)
top30tfidf = tfidf.get_top_n_tfidf_bow(selected_tokens, top_n_tokens = 30)
print('top 30 tfidf', top30tfidf)
no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(selected_tokens, num_topics = no_topics)
lda.lda_topics(lda_allbow)


# In[44]:


DataExploration.generate_word_cloud(selected_tokens)


# ## Fit model based on title

# In[58]:


train_index, test_index= train_test_split(train.index , test_size = 0.33, random_state = 42)
X_train = train[train.index.isin(train_index)][['TITLE']]
X_test = train[train.index.isin(test_index)][['TITLE']]
y_train = pd.get_dummies(train[train.index.isin(train_index)]['LABEL']).values
y_test = pd.get_dummies(train[train.index.isin(test_index)]['LABEL']).values
X_train = [i for i in X_train.TITLE]
X_test = [i for i in X_test.TITLE]


# In[60]:


train['t_len'] = train.TITLE.apply(lambda x: len(x))
max_length = int(train.t_len.quantile(0.99))
max_length


# In[61]:


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


# In[79]:


model = Sequential(
    [
        Embedding(vocab_size, 100, weights = [embedding_matrix], input_length = max_length, trainable = False),
        Flatten(),
        Dense(100, activation="relu", name="layer1"),
        Dense(6, activation = 'softmax', name="layer2")
        
    ]
)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
print(model.summary())


# In[80]:


# fit the model
history = model.fit(padded_docs, 
                    y_train, 
                    epochs = 5, 
                    verbose = 1, 
                    batch_size = 32, 
                    validation_data = (padded_test_docs, y_test)
                   )


# In[81]:


class_names = ['Computer Science','Physics','Mathematics','Statistics','Quantitative Biology','Quantitative Finance']
predsTest = pd.DataFrame(model.predict(padded_test_docs), columns = class_names)
predsTest['pred'] = predsTest.idxmax(axis = 1)
predsTest.head(3)


# In[83]:


roundedPredsTest = predsTest.pred
print('Confusion Matrix: ')
cf_matrix = confusion_matrix(val[val.index.isin(test_index)]['LABEL'], roundedPredsTest, labels = class_names)
print(cf_matrix)
meplot.cf_matrix_heatmap(cf_matrix, class_names, 13)


# In[84]:


print(classification_report(val[val.index.isin(test_index)]['LABEL'], roundedPredsTest))

