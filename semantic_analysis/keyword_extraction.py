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


text_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data\\BBC_News_Summary\\BBC_News_Summary\\News_Articles'
smr_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data\\BBC_News_Summary\\BBC_News_Summary\\Summaries'
# https://www.kaggle.com/datajameson/topic-modelling-nlp-amazon-reviews-bbc-news
# reference: https://towardsdatascience.com/keyword-extraction-python-tf-idf-textrank-topicrank-yake-bert-7405d51cd839


# In[2]:


folders=["business","entertainment","politics","sport","tech"]
x=[]
y=[]
z=[]

for i in folders:
    files=os.listdir(os.path.join(text_path, i))
    for text_file in files:
        file_path=os.path.join(os.path.join(text_path, i), text_file)
        with open(file_path,'rb') as f:
            data=f.read().decode('iso-8859-1')
        x.append(data)
        y.append(i)
        z.append(i+text_file[:3])
        
data={'news':x,'type':y, 'docid':z}
textdf = pd.DataFrame(data)

folders=["business","entertainment","politics","sport","tech"]
x=[]
y=[]
z=[]

for i in folders:
    files=os.listdir(os.path.join(smr_path, i))
    for text_file in files:
        file_path=os.path.join(os.path.join(smr_path, i), text_file)
        with open(file_path,'rb') as f:
            data=f.read()
        x.append(data)
        y.append(i)
        z.append(i+text_file[:3])
        
data={'news':x,'type':y, 'docid':z}
smrdf = pd.DataFrame(data)


# In[4]:


rawdata = textdf.merge(smrdf, how='left',on=['docid','type']).rename(columns={"news_x": "news", "news_y": "summary"})
rawdata


# In[5]:


def print_article(data, index_id):
    printdata = data[data.index==index_id]
    print('Type:',printdata.type.item(), '// docid:', printdata.docid.item())
    print('-------------------- Summary --------------------')
    print('Description:',printdata.summary.item())
    print('-------------------- News --------------------')
    print(printdata.news.item())

print_article(rawdata,0)


# ## TF-IDF

# In[21]:


# Generate key words from multiple documents
preprocessed_tokens = textClean.pipeline(rawdata[rawdata.index.isin(range(10))]['news'], multi_gram = [1], lower_case=True, deacc=False, encoding='utf8',
                                         errors='strict', stem_lemma = 'lemma', tag_drop = [], nltk_stop=True, stop_word_list=[], 
                                         check_numbers=True, word_length=3, remove_consecutives=True)
tfidf.get_top_n_tfidf_bow(preprocessed_tokens, top_n_tokens = 10, no_below =1)


# In[22]:


# Generate key words from 1 document
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(rawdata[rawdata.index.isin([0])]['news'])
names = vectorizer.get_feature_names()
data = vectors.todense().tolist()
df = pd.DataFrame(data, columns=names)

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
st = set(stopwords.words('english'))
df = df[filter(lambda x: x not in list(st) , df.columns)]

N = 10;
for i in df.iterrows():
    print(i[1].sort_values(ascending=False)[:N])


# In[37]:


preprocessed_tokens


# ## TextRank

# In[39]:


from summa import keywords
keywords.keywords(rawdata[rawdata.index.isin([0])]['news'][0].translate('0123456789'), words=10).split("\n")


# In[33]:


rawdata[rawdata.index.isin([0])]['news'][0]


# ## Topic Rank

# In[40]:


# pytopicrank version is not match
from pytopicrank import TopicRank
tr = TopicRank(rawdata[rawdata.index.isin([0])]['news'][0])
tr.get_top_n(n=10, extract_strategy='first')


# ## YAKE

# In[42]:


from yake import KeywordExtractor

kw_extractor = KeywordExtractor(lan="en", n=1, top=10)
keywords = kw_extractor.extract_keywords(text=rawdata[rawdata.index.isin([0])]['news'][0])
keywords = [x for x, y in keywords]
print(keywords)


# ## BERT

# In[48]:


from keybert import KeyBERT
kw_extractor = KeyBERT('distilbert-base-nli-mean-tokens')
keywords = kw_extractor.extract_keywords(rawdata[rawdata.index.isin([0])]['news'][0], 
                                         keyphrase_ngram_range=(1, 2),
                                         stop_words='english',
                                         top_n=10)
print(keywords)

