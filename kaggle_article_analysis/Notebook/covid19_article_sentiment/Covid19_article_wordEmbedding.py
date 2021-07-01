#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


raw_data = pd.read_csv(os.path.join(root_path, "covid-19_articles_data.csv"))


# In[4]:


raw_data['len'] = raw_data.text.apply(lambda x: len(x.split()))


# In[5]:


raw_data.len.describe()


# In[6]:


raw_data.sentiment.value_counts()


# In[7]:


print(raw_data.text[0])


# In[8]:


print(raw_data[raw_data.len==30].text)


# In[10]:


raw_data = raw_data[['text','sentiment','len']]
raw_data['doc_id'] = raw_data.index
train_index, test_index= train_test_split(raw_data['doc_id'] , test_size = 0.33, random_state = 42)
train_index.head(3)


# In[11]:


raw_data.head(3)


# ## TFIDF / XGBoost

# In[13]:


data = raw_data.copy()


# In[15]:


data['token'] = textClean.pipeline(raw_data['text'].to_list(), multi_gram = [1], lower_case=True, deacc=False, encoding='utf8',
                                   errors='strict', stem_lemma = 'lemma', tag_drop = [], nltk_stop=True, 
                                   stop_word_list=[], remove_pattern = [],
                                   check_numbers=True, word_length=2, remove_consecutives=True)


# In[16]:


top_10_freq_words = [i[0] for i in DataExploration.get_topn_freq_bow(data['token'].to_list(), topn = 10)]
print(top_10_freq_words)


# In[17]:


top30tfidf = tfidf.get_top_n_tfidf_bow(data['token'].to_list(), top_n_tokens = 30)
print('top 30 tfidf', top30tfidf)


# In[18]:


DataExploration.generate_word_cloud(data['token'].to_list())


# In[19]:


no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(data['token'].to_list(), num_topics = no_topics)
lda.lda_topics(lda_allbow)


# In[33]:


tfidf_data = tfidf.get_tfidf_dataframe(data['token'].to_list(), 
                                       doc_index = list(data.doc_id),
                                       no_below =5, 
                                       no_above = 0.5, 
                                       keep_n = 100000)


# In[28]:


tfidf_data.columns = ['doc_id'] + [i[1] for i in tfidf_data.columns][1:]
tfidf_data.head(3)


# In[34]:


tfidf_data = tfidf_data.pivot(index=['doc_id'], columns='bow').fillna(0).reset_index()
tfidf_data.columns = ['doc_id'] + [i[1] for i in tfidf_data.columns][1:]
tfidf_data.head(3)


# In[35]:


tfidf_data['sentiment'] = data.sentiment


# In[36]:


X_train = tfidf_data[tfidf_data.doc_id.isin(train_index)].drop(columns = ['doc_id','sentiment'])
X_test = tfidf_data[tfidf_data.doc_id.isin(test_index)].drop(columns = ['doc_id','sentiment'])
y_train = tfidf_data[tfidf_data.doc_id.isin(train_index)][['sentiment']]
y_test = tfidf_data[tfidf_data.doc_id.isin(test_index)][['sentiment']]


# In[39]:


import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label = y_train, feature_names = X_train.columns)
dtest = xgb.DMatrix(X_test,  label = y_test,  feature_names = X_test.columns)
evallist = [(dtrain,'train'), (dtest,'validation')]
# specify parameters via map
param = {'objective':'binary:logistic',
         'max_depth':6,
         'learning_rate': 0.3,
         'eval_metric': 'auc',
         'subsample': 1,
         'colsample_bytree': 1,
         'colsample_bylevel': 1,
         'reg_alpha': 0,                  # 0.0  (L1 regularization)
         'reg_lambda': 1,                 # 1.0  (L2 regularization)
         'min_split_loss': 0,
         'min_child_weight': 1,
         'silent': 1
        }

num_round = 100
early_stop_round = 50
train_monitor = dict()
xgbmodel = xgb.train(param, 
                     dtrain, 
                     num_boost_round = num_round,
                     evals = evallist, 
                     verbose_eval = 10,
                     early_stopping_rounds = early_stop_round,
                     evals_result = train_monitor
                    )


# In[43]:


#Check Model Performance on the test set
y_pred = xgbmodel.predict(dtest)
meplot.plot_randomfp_roc(X_train, y_train, X_test, y_test, xgbmodel.predict(dtest), model_label = 'XGBoost')    


# In[45]:


cf_matrix = confusion_matrix(y_test, np.round(y_pred))
meplot.cf_matrix_heatmap(cf_matrix)


# In[46]:


variable_importance = np.array(list(xgbmodel.get_score(importance_type='gain').values()))
meplot.plot_var_imp(variable_importance, X_train.columns, 20)


# In[47]:


print(classification_report(y_test, np.round(y_pred)))


# ## Word Embedding / RNN

# In[80]:


X_train = data[data.index.isin(train_index)][['text']]
X_test = data[data.index.isin(test_index)][['text']]
y_train = data[data.index.isin(train_index)]['sentiment']
y_test = data[data.index.isin(test_index)]['sentiment']
X_train = [i for i in X_train.text]
X_test = [i for i in X_test.text]


# In[70]:


max_length = int(raw_data.len.quantile(0.99))
max_length


# In[81]:


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


# In[82]:


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


# In[83]:


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


# #### RNN

# In[84]:


model = Sequential(
    [
        Embedding(vocab_size, 100, weights = [embedding_matrix], input_length = max_length, trainable = False),
        Flatten(),
        Dense(100, activation="relu", name="layer1"),
        Dense(1, activation = 'sigmoid', name="layer2")
    ]
)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
print(model.summary())


# In[85]:


# fit the model
history = model.fit(padded_docs, 
                    y_train, 
                    epochs = 10, 
                    verbose = 1, 
                    batch_size = 32, 
                    validation_data = (padded_test_docs, y_test)
                   )


# In[86]:


predsTest = model.predict(padded_test_docs)
roundedPredsTest = np.round(predsTest)
print('Confusion Matrix: Positive is class 1 and Negative is class 0')
cf_matrix = confusion_matrix(y_test, roundedPredsTest, labels = [1,0])
print(cf_matrix)
meplot.cf_matrix_heatmap(cf_matrix)


# In[87]:


print(classification_report(y_test,roundedPredsTest))


# In[90]:


from tensorflow.keras.layers import LSTM
lstm = Sequential(
    [
        Embedding(vocab_size, 100, weights = [embedding_matrix], input_length = max_length, trainable = False),
        LSTM(100, dropout=0.2, recurrent_dropout=0.2, name = 'lstm'),
        Dense(1, activation = 'sigmoid', name="layer2")
    ]
)
lstm.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
print(lstm.summary())


# In[91]:


# fit the model
history = lstm.fit(padded_docs, 
                    y_train, 
                    epochs = 10, 
                    verbose = 1, 
                    batch_size = 32, 
                    validation_data = (padded_test_docs, y_test)
                   )


# In[93]:


predsTest = lstm.predict(padded_test_docs)
roundedPredsTest = np.round(predsTest)
print('Confusion Matrix: Positive is class 1 and Negative is class 0')
cf_matrix = confusion_matrix(y_test, roundedPredsTest, labels = [1,0])
print(cf_matrix)
meplot.cf_matrix_heatmap(cf_matrix)


# In[94]:


print(classification_report(y_test,roundedPredsTest))


# From the above analysis we can see the tfidf+xgboost performance is better than word-embedding + rnn/lstm.

# ## AWD_LSTM
# Using fastai in google colab Covid19_article_wordEmbedding_gcolab.ipynb

# ## Contextual string embeddings

# ## Transformers
