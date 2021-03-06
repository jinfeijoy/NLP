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


# In[3]:


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


# ## LSA

# In[6]:


lsa_smr = txtsmr.lsa_text_extraction(rawdata.news[0], smooth = 0.4, topn = 6)
print("\n".join(lsa_smr))


# ## Glove similarity

# In[7]:


embeddings_index = wdembd.loadGloveModel('C:/ProgramData/Anaconda3/append_file/glove/glove.6B.100d.txt')
glv_smr = txtsmr.embedding_similarity_pagerank_extraction(rawdata.news[0], 100, embeddings_index)
print("\n".join(glv_smr))


# ## Package sumy

# In[8]:


import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


# In[9]:


parser = PlaintextParser.from_string(rawdata.news[1], Tokenizer("english"))
stemmer = Stemmer("english")
summarizer = Summarizer(stemmer)
summarizer.stop_words = get_stop_words("english")

for sentence in summarizer(parser.document, 6):
    print(sentence)


# In[10]:


testchinese = '?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????'
parser = PlaintextParser.from_string(testchinese, Tokenizer("chinese"))
stemmer = Stemmer("chinese")
summarizer = Summarizer(stemmer)
summarizer.stop_words = get_stop_words("chinese")

for sentence in summarizer(parser.document, 6):
    print(sentence)


# In[ ]:




