#!/usr/bin/env python
# coding: utf-8

# In[44]:


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
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint

datapath = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data'


# In[3]:


data = pd.read_csv(os.path.join(datapath,'ner_dataset.csv'), encoding= 'unicode_escape')
data.columns = ['sentence', 'word', 'pos', 'tag']
data.ffill(inplace=True)
data.head()


# In[41]:


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


# In[45]:


def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent
tree2conlltags(preprocess(resume_df.iloc[0]['content']))


# ## NLTK

# In[56]:


sample = data[data.sentence == 'Sentence: 3']
pattern = 'NP: {<DT>?<JJ>*<NN>}'
cp = nltk.RegexpParser(pattern)
cs = cp.parse(dict(zip(sample.word, sample.pos)).items())
print(cs)


# In[47]:


iob_tagged = tree2conlltags(cs)
pprint(iob_tagged)
pprint(sample)


# In[48]:


ne_tree = nltk.ne_chunk(list(zip(sample.word, sample.pos)))
print(ne_tree)


# ## SpaCy

# In[49]:


import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()


# In[63]:


sample = data[data.sentence == 'Sentence: 8']
doc = nlp(' '.join(sample.word))
pprint([(X.text, X.label_) for X in doc.ents])
pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])


# In[64]:


displacy.render(doc, jupyter=True, style='ent')
displacy.render(doc, style='dep', jupyter = True, options = {'distance': 120})


# In[52]:


[(x.orth_,x.pos_, x.lemma_) for x in [y 
                                      for y
                                      in doc 
                                      if not y.is_stop and y.pos_ != 'PUNCT']]

