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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import re
import seaborn as sns

import nltk

datapath = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data'


# In[2]:


data = pd.read_csv(os.path.join(datapath,'ner_dataset.csv'), encoding= 'unicode_escape')
data.columns = ['sentence', 'word', 'pos', 'tag']
data.ffill(inplace=True)
data.head()


# ## NLTK

# In[3]:


sample = data[data.sentence == 'Sentence: 1']
pattern = 'NP: {<DT>?<JJ>*<NN>}'
cp = nltk.RegexpParser(pattern)
cs = cp.parse(dict(zip(sample.word, sample.pos)).items())
print(cs)


# In[4]:


from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
iob_tagged = tree2conlltags(cs)
pprint(iob_tagged)
pprint(sample)


# In[5]:


ne_tree = nltk.ne_chunk(list(zip(sample.word, sample.pos)))
print(ne_tree)


# ## SpaCy

# In[6]:


import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()


# In[11]:


doc = nlp(' '.join(sample.word))
pprint([(X.text, X.label_) for X in doc.ents])
pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])


# In[13]:


displacy.render(doc, jupyter=True, style='ent')
displacy.render(doc, style='dep', jupyter = True, options = {'distance': 120})


# In[14]:


[(x.orth_,x.pos_, x.lemma_) for x in [y 
                                      for y
                                      in doc 
                                      if not y.is_stop and y.pos_ != 'PUNCT']]

