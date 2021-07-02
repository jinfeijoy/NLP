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

import matplotlib
import matplotlib.dates as mdates
root_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data'


# ## Load Data and Data Visualization

# In[2]:


raw_data = pd.read_csv(os.path.join(root_path, "fake_job_postings.csv"))
raw_data['jd'] = raw_data[['title','function','employment_type','company_profile','description','requirements','benefits']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
raw_data.head(3)


# Columns `telecommuting`, `has_company_logo`, `has_questions`, `employment_type`, `required_experience`, `required_education` and `function` are clean and we will visualize these columns. All other columns are messy or text columns, will be ignored or discussed later.

# In[78]:


raw_data['location_country'] = raw_data.location.str[:2]


# In[79]:


dbplot.plot_count_dist("location_country", "location_country", raw_data, 4, True)


# In[89]:


dbplot.generate_bar_proportion(raw_data, 'fraudulent', 'location_country', color = 0, order = True, topn = 50)


# In[17]:


dbplot.plot_count_dist("function", "function", raw_data, 4, True)


# In[43]:


dbplot.generate_bar_proportion(raw_data, 'fraudulent', 'function', color = 0, order = True, topn = 10)


# In[28]:


dbplot.plot_count_dist("required_education", "required_education", raw_data, 4, True)


# In[45]:


dbplot.generate_bar_proportion(raw_data, 'fraudulent', 'required_education', color = 0, order = True, topn = 10)


# In[46]:


dbplot.plot_count_dist("employment_type", "employment_type", raw_data, 4, True)


# In[47]:


dbplot.generate_bar_proportion(raw_data, 'fraudulent', 'employment_type', color = 0, order = True, topn = 12)


# In[48]:


dbplot.plot_count_dist("required_experience", "required_experience", raw_data, 4, True)


# In[58]:


dbplot.generate_bar_proportion(raw_data, 'fraudulent', 'required_experience', color = 0, order = True, topn = 12)


# In[63]:


binary_dic = {1:'1', 0:'0'}
raw_data['telecommuting'] = raw_data['telecommuting'].replace(binary_dic)
raw_data['has_company_logo'] = raw_data['has_company_logo'].replace(binary_dic)
raw_data['has_questions'] = raw_data['has_questions'].replace(binary_dic)


# In[64]:


dbplot.plot_count_dist("telecommuting", "telecommuting", raw_data, 4, True)


# In[65]:


dbplot.generate_bar_proportion(raw_data, 'fraudulent', 'telecommuting', color = 0, order = True, topn = 12)


# In[66]:


dbplot.plot_count_dist("has_company_logo", "has_company_logo", raw_data, 4, True)


# In[67]:


generate_bar_proportion(raw_data, 'fraudulent', 'has_company_logo', color = 0, order = True, topn = 12)


# In[68]:


dbplot.plot_count_dist("has_questions", "has_questions", raw_data, 4, True)


# In[69]:


generate_bar_proportion(raw_data, 'fraudulent', 'has_questions', color = 0, order = True, topn = 12)


# ## Text Exploration and Visualization

# In[136]:


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


# In[145]:


# raw_data['jd_tokens'] = textClean.pipeline(raw_data['jd'].to_list(), multi_gram = [1], lower_case=True, 
#                                            deacc=False, encoding='utf8', errors='strict', stem_lemma = 'lemma', 
#                                            tag_drop = [], nltk_stop=True, 
#                                            stop_word_list=[], 
#                                            check_numbers=False, word_length=2, remove_consecutives=True)
fraud_tokens = list(raw_data[raw_data.fraudulent==1]['jd_tokens'])
print(tfidf.get_top_n_tfidf_bow(fraud_tokens, top_n_tokens = 30))
DataExploration.generate_word_cloud(fraud_tokens)
no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(fraud_tokens, num_topics = 10)
lda.lda_topics(lda_allbow)


# In[146]:


raw_data['jd_tokens2'] = textClean.pipeline(raw_data['jd'].to_list(), multi_gram = [2], lower_case=True, 
                                           deacc=False, encoding='utf8', errors='strict', stem_lemma = 'lemma', 
                                           tag_drop = [], nltk_stop=True, 
                                           stop_word_list=[], 
                                           check_numbers=False, word_length=2, remove_consecutives=True)
print(tfidf.get_top_n_tfidf_bow(list(raw_data['jd_tokens2']), top_n_tokens = 30))
DataExploration.generate_word_cloud(list(raw_data['jd_tokens2']))
no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(list(raw_data['jd_tokens2']), num_topics = 10)
lda.lda_topics(lda_allbow)

