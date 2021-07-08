#!/usr/bin/env python
# coding: utf-8

# In[10]:


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
import pycountry
import seaborn as sns
import plotly.express as px
from gensim.models.coherencemodel import CoherenceModel

import matplotlib
import matplotlib.dates as mdates
root_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data'


# ## Load Data and Data Visualization

# In[2]:


raw_data = pd.read_csv(os.path.join(root_path, "fake_job_postings.csv"))
raw_data['jd'] = raw_data[['title','function','employment_type','company_profile','description','requirements','benefits']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
raw_data.head(3)


# In[3]:


def convert_str(data, cols):
    df = data.copy()
    for i in cols:
        df[i] = df[i].apply(lambda x: str(x))
    return df
raw_data = convert_str(raw_data, ['company_profile','description','requirements','benefits'])


# In[4]:


raw_data['profile_tokens'] = textClean.pipeline(raw_data['company_profile'].to_list(), multi_gram = [1], lower_case=True, 
                                           deacc=False, encoding='utf8', errors='strict', stem_lemma = 'lemma', 
                                           tag_drop = [], nltk_stop=True, 
                                           stop_word_list=[], 
                                           check_numbers=False, word_length=2, remove_consecutives=True)
raw_data['description_tokens'] = textClean.pipeline(raw_data['description'].to_list(), multi_gram = [1], lower_case=True, 
                                           deacc=False, encoding='utf8', errors='strict', stem_lemma = 'lemma', 
                                           tag_drop = [], nltk_stop=True, 
                                           stop_word_list=[], 
                                           check_numbers=False, word_length=2, remove_consecutives=True)
raw_data['requirements_tokens'] = textClean.pipeline(raw_data['requirements'].to_list(), multi_gram = [1], lower_case=True, 
                                           deacc=False, encoding='utf8', errors='strict', stem_lemma = 'lemma', 
                                           tag_drop = [], nltk_stop=True, 
                                           stop_word_list=[], 
                                           check_numbers=False, word_length=2, remove_consecutives=True)
raw_data['benefit_tokens'] = textClean.pipeline(raw_data['benefits'].to_list(), multi_gram = [1], lower_case=True, 
                                           deacc=False, encoding='utf8', errors='strict', stem_lemma = 'lemma', 
                                           tag_drop = [], nltk_stop=True, 
                                           stop_word_list=[], 
                                           check_numbers=False, word_length=2, remove_consecutives=True)


# In[6]:


def print_job(data, jobid):
    printdata = data[data.job_id==jobid]
    print('Title:',printdata.title.item(), '/ Location:', printdata.location.item())
    print('Department:',printdata.department.item(), '/ Salary:', printdata.salary_range.item(), '/ Employment type:', printdata.employment_type.item())
    print('Function:',printdata.function.item(), '/ Required Experience:', printdata.required_experience.item(), '/ Required Education:', printdata.required_education.item())
    print('Fraudulent:', printdata.fraudulent.item())
    print('-------------------- Company --------------------')
    print(printdata.company_profile.item())
    print('-------------------- Job Description --------------------')
    print(printdata.description.item())
    print('-------------------- Requirements --------------------')
    print(printdata.requirements.item())
    print('-------------------- Benifits --------------------')
    print(printdata.benefits.item())
print_job(raw_data,50)


# In[5]:


profile_tokens = list(raw_data['profile_tokens'])
print(tfidf.get_top_n_tfidf_bow(profile_tokens, top_n_tokens = 30))
DataExploration.generate_word_cloud(profile_tokens)
no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(profile_tokens, num_topics = 10)
lda.lda_topics(lda_allbow)


# In[6]:


profile_tokens = list(raw_data['description_tokens'])
print(tfidf.get_top_n_tfidf_bow(profile_tokens, top_n_tokens = 30))
DataExploration.generate_word_cloud(profile_tokens)
no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(profile_tokens, num_topics = 10)
lda.lda_topics(lda_allbow)


# In[7]:


profile_tokens = list(raw_data['requirements_tokens'])
print(tfidf.get_top_n_tfidf_bow(profile_tokens, top_n_tokens = 30))
DataExploration.generate_word_cloud(profile_tokens)
no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(profile_tokens, num_topics = 10)
lda.lda_topics(lda_allbow)


# In[24]:


profile_tokens = list(raw_data['benefit_tokens'])
print(tfidf.get_top_n_tfidf_bow(profile_tokens, top_n_tokens = 30))
DataExploration.generate_word_cloud(profile_tokens)
no_topics = 5
lda_allbow, bow_corpus, dictionary = lda.fit_lda(profile_tokens, num_topics = 10)
print('Coherence',CoherenceModel(model=lda_allbow,texts=profile_tokens,dictionary=dictionary,coherence='c_v').get_coherence())
lda.lda_topics(lda_allbow)


# In[9]:


import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
pyLDAvis.enable_notebook()
vis = gensimvis.prepare(lda_allbow, bow_corpus, dictionary)
vis


# ## lda2vec
# https://multithreaded.stitchfix.com/blog/2016/05/27/lda2vec/#topic=5&lambda=1&term=
# 
#  At a practical level, if you want human-readable topics just use LDA (checkout libraries in scikit-learn and gensim). If you want machine-useable word-level features, use word2vec. But if you want to rework your own topic models that, say, jointly correlate an article’s topics with votes or predict topics over users then you might be interested in lda2vec.
