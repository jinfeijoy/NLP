#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
import data_visualization.distribution_plot as dbplot

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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
from nltk.tokenize import word_tokenize
pd.set_option('display.max_colwidth', None)
root_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data'


# In[5]:


raw_data = pd.read_csv(os.path.join(root_path, "fake_job_postings.csv"))
raw_data['jd'] = raw_data[['title','function','employment_type','company_profile','description','requirements','benefits']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
raw_data.head(1)


# In[6]:


df_train, df_val = train_test_split(raw_data, test_size=0.33, random_state=42)


# In[7]:


tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(raw_data['jd'].to_list())]
max_epochs = 100
vec_size = 100
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
model.save(os.path.join(root_path, "d2v.model"))


# In[9]:


df_train, df_test = train_test_split(raw_data, test_size=0.3, random_state=42)
model= Doc2Vec.load(os.path.join(root_path, "d2v.model"))
train_tag = df_train.apply(lambda r: TaggedDocument(words=word_tokenize(r['jd']), tags=[r.fraudulent]), axis=1)
test_tag = df_test.apply(lambda r: TaggedDocument(words=word_tokenize(r['jd']), tags=[r.fraudulent]), axis=1)
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in tagged_docs])
    return targets, regressors


# In[10]:


y_train, X_train = vec_for_learning(model, train_tag)
y_test, X_test = vec_for_learning(model, test_tag)


# In[11]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)


# In[12]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[15]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

