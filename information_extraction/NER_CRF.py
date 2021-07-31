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
import nlpbasic.text_summarize as txtsmr
import nlpbasic.word_embedding as wdembd

import model_explain.plot as meplot
import model_explain.shap as meshap
import data_visualization.distribution_plot as dbplot

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import re
import seaborn as sns
datapath = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data'


# ## Load Data
# **NER dataset with word tag**

# In[2]:


data = pd.read_csv(os.path.join(datapath,'ner_dataset.csv'), encoding= 'unicode_escape')
data.columns = ['sentence', 'word', 'pos', 'tag']
data.ffill(inplace=True)
data.head()


# In[3]:


dbplot.plot_count_dist("tag", "tag", data[data.tag!='O'], 4, True)


# In[4]:


data.info()


# **Resume json data**

# In[ ]:


import json
from collections import defaultdict

def pop_annot(raw_line):
    in_line = defaultdict(list, **raw_line)
    if 'annotation' in in_line:
        labels = in_line['annotation']
        for c_lab in labels:
            if len(c_lab['label'])>0:
                in_line[c_lab['label'][0]] += c_lab['points']
    return in_line
with open(os.path.join(datapath, 'Entity Recognition in Resumes.json'), encoding="utf8") as f:
    # data is jsonl and so we parse it line-by-line
    resume_data = [json.loads(f_line) for f_line in f.readlines()]
    resume_df = pd.DataFrame([pop_annot(line) for line in resume_data])
    
for col in resume_df.columns:
    print(col)
    display(resume_df.iloc[0][col])
    
displacy.render(nlp(resume_df.iloc[0]['content']), jupyter=True, style='ent')


# ## CRF
# Reference: https://www.kaggle.com/shoumikgoswami/ner-using-random-forest-and-crf

# In[5]:


def feature_map(word):
    return np.array([word.istitle(), word.islower(), word.isupper(), len(word),
                     word.isdigit(),  word.isalpha()])
feature_map(data.iloc[0]['word'])


# In[6]:


class getsentence(object):
    
    def __init__(self, data):
        self.n_sent = 1.0
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["word"].values.tolist(),
                                                           s["pos"].values.tolist(),
                                                           s["tag"].values.tolist())]
        self.grouped = self.data.groupby("sentence").apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
getter = getsentence(data)
sentences = getter.sentences
sentences[0]


# In[21]:


from itertools import chain

import nltk
import sklearn
import scipy.stats

import sklearn_crfsuite
from sklearn_crfsuite import scorers,CRF
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn_crfsuite import metrics
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer


# In[8]:


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]


# In[9]:


X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]


# In[29]:


X[0]


# In[30]:


y[0]


# In[17]:


crf = CRF(algorithm='lbfgs',
          c1=0.1,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)
pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)
report = flat_classification_report(y_pred=pred, y_true=y)
print(report)


# In[18]:


#Tuning the parameters manually, setting c1 = 10
crf2 = CRF(algorithm='lbfgs',
          c1=10,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)
pred = cross_val_predict(estimator=crf2, X=X, y=y, cv=5)
report = flat_classification_report(y_pred=pred, y_true=y)
print(report)
crf2.fit(X, y)


# In[19]:


labels = list(crf2.classes_)
labels = list(filter(lambda a: a != 'O', labels))
print(labels)


# In[22]:


crf3 = CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True
)
params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),
}

# use the same metric for evaluation
f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=labels)

# search
rs = RandomizedSearchCV(crf, params_space,
                        cv=3,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=50,
                        scoring=f1_scorer)
rs.fit(X, y)


# In[23]:


print('Best parameters:', rs.best_params_)
print('Best CV score:', rs.best_score_)
print('Model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))


# In[24]:


sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0]))


# In[25]:


#Now we create the model again using the best estimators
crf3 = rs.best_estimator_
y_pred = crf3.predict(X)
print(metrics.flat_classification_report(
    y, y_pred, labels=sorted_labels, digits=3))


# In[26]:


crf3.fit(X,y)


# In[27]:


import eli5
eli5.show_weights(crf3, top=30)


# In[34]:


import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))


# In[ ]:


get_ipython().system('pip install -U scikit-learn==0.23.2')

