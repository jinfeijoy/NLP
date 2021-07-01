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
root_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data'


# In[54]:


raw_data = pd.read_csv(os.path.join(root_path, "covid_tweet_sentiment_after_fastai.csv"))
raw_data = raw_data[raw_data.orig_text.isnull()==False]
raw_data = raw_data.drop_duplicates()
raw_data.hashtags = raw_data.hashtags.apply(lambda x: str(x).lower())
print(len(raw_data))
raw_data.head(3)


# In[23]:


test = raw_data[['label','sentiment','orig_text','id','created_at','favorite_count','retweet_count','hashtags','place']]
test['label'] = test['label'].apply(lambda x: x[0:3])
test['label_comp'] = np.where(test["sentiment"] == test["label"], True, False)
test['date'] = pd.to_datetime(test['created_at'], errors='coerce').dt.date
test.head(3)


# In[22]:


i =  7 
print(test.orig_text[i])
print(test.label[i])
print(test.sentiment[i])


# In[25]:


timeline = test.groupby(['date', 'sentiment']).agg(**{'tweets': ('id', 'count')}).reset_index().dropna()
timeline.head(3)


# In[26]:


import plotly.express as px
fig = px.line(timeline, x='date', y='tweets', color='sentiment', category_orders={'sentiment': ['neutral', 'negative', 'positive']},
             title='Timeline showing sentiment of tweets about COVID-19 vaccines')
fig.show()


# In[43]:


spike = test[test['date'].astype(str)=='2020-09-27']


# In[44]:


spike.place.value_counts(ascending=False).head(20)


# Malaysia pop-up, which is abnormal, so we need to explore what happened in Malaysia that day.

# In[46]:


pd.set_option('display.max_colwidth', None)
malaysia_test = spike[spike['place'] == 'Malaysia']
print(malaysia_test.sentiment.value_counts())
malaysia_test.orig_text.head(10)


# From above, we can see covid situation was getting worse in 2020Sep27 in Malaysia, and the tweets are mostly neutral.

# In[55]:


raw_data.hashtags.value_counts().head(25)

