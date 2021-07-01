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


# In[6]:


def de_emojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')
def tweet_proc(df, text_col='text'):
    df['orig_text'] = df[text_col]
    # Remove twitter handles
    df[text_col] = df[text_col].apply(lambda x:re.sub('@[^\s]+','',x))
    # Remove URLs
    df[text_col] = df[text_col].apply(lambda x:re.sub(r"http\S+", "", x))
    # Remove emojis
    df[text_col] = df[text_col].apply(de_emojify)
    # Remove hashtags
    df[text_col] = df[text_col].apply(lambda x:re.sub(r'\B#\S+','',x))
    return df[df[text_col]!='']


# In[76]:


raw_data = pd.read_csv(os.path.join(root_path, "covidvacc_tweet_sentiment_emo.csv"))


# In[77]:


len(raw_data)


# In[78]:


data = raw_data[raw_data.text.isnull()==False]
data = data.drop_duplicates()
len(data)


# In[79]:


data.head(3)


# In[80]:


data = tweet_proc(data,'text')
data['date']=pd.to_datetime(data['date'], errors='coerce').dt.date


# In[8]:


i = 3
print(data.text[i])
print("-----------")
print(data.orig_text[i])


# ## Vader
# Apply vader and generate neg, neu, pos and compound. This dataset already have these columns, this analysis can be done in the future task. Reference: https://predictivehacks.com/how-to-run-sentiment-analysis-in-python-using-vader/

# In[10]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import twitter_samples


# In[54]:


vader_analyzer = SentimentIntensityAnalyzer()
vader_analyzer.polarity_scores(data.text[0])
# data['neg'] = data['clean_tweet'].apply(lambda x: vader_analyzer.polarity_scores(x)['neg'])
# data['neu'] = data['clean_tweet'].apply(lambda x: vader_analyzer.polarity_scores(x)['neu'])
# data['pos'] = data['clean_tweet'].apply(lambda x: vader_analyzer.polarity_scores(x)['pos'])
data['compound'] = data['text'].apply(lambda x: vader_analyzer.polarity_scores(x)['compound'])


# In[55]:


data.head(3)


# ## Explore and Visualization

# In[56]:


data['explore_text'] = textClean.pipeline(data['text'].to_list(), multi_gram = [1], lower_case=True, deacc=False, encoding='utf8',
                                          errors='strict', stem_lemma = 'lemma', tag_drop = [], nltk_stop=True, 
                                          stop_word_list=[], 
                                          check_numbers=False, word_length=2, remove_consecutives=True)


# In[57]:


i = 1
print(data.orig_text[i])
print("-----------")
print(data.text[i])
print("-----------")
print(data.explore_text[i])


# In[58]:


pos_tweet = list(data[data['compound']>0]['explore_text'])
neg_tweet = list(data[data['compound']<0]['explore_text'])
neu_tweet = list(data[data['compound']==0]['explore_text'])


# In[59]:


postop10tfidf = tfidf.get_top_n_tfidf_bow(pos_tweet, top_n_tokens = 30)
negtop10tfidf = tfidf.get_top_n_tfidf_bow(neg_tweet, top_n_tokens = 30)
neutop10tfidf = tfidf.get_top_n_tfidf_bow(neu_tweet, top_n_tokens = 30)
print('top 30 negative review tfidf', negtop10tfidf)
print('top 30 positive review tfidf', postop10tfidf)
print('top 30 neutual review tfidf', neutop10tfidf)


# In[40]:


top10_posfreq_list = DataExploration.get_topn_freq_bow(pos_tweet, topn = 10)
top10_negfreq_list = DataExploration.get_topn_freq_bow(neg_tweet, topn = 10)
print(top10_posfreq_list)
print(top10_negfreq_list)


# In[38]:


DataExploration.generate_word_cloud(pos_tweet)


# In[41]:


DataExploration.generate_word_cloud(neg_tweet)


# In[42]:


DataExploration.generate_word_cloud(neu_tweet)


# ## LDA

# In[43]:


no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(pos_tweet, num_topics = no_topics)
# lda_top30bow, bow_corpus, dictionary  = lda.fit_lda(pos_tweet, top_n_tokens = 30, num_topics = no_topics)
lda.lda_topics(lda_allbow)


# In[44]:


no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(neg_tweet, num_topics = no_topics)
# lda_top30bow, bow_corpus, dictionary  = lda.fit_lda(neg_tweet, top_n_tokens = 30, num_topics = no_topics)
lda.lda_topics(lda_allbow)


# ## Explore text for each vacc

# In[51]:


all_vax = ['covaxin', 'sinopharm', 'sinovac', 'moderna', 'pfizer', 'biontech', 'oxford', 'astrazeneca', 'sputnik']

# Note: a lot of the tweets seem to contain hashtags for multiple vaccines even though they are specifically referring to one vaccine - not very helpful!
def filtered_vacc(df, vax):
    df = df.dropna()
    df_filt = pd.DataFrame()
    for o in vax:
        df_filt = df_filt.append(df[df['text'].str.lower().str.contains(o)])
    other_vax = list(set(all_vax)-set(vax))
    for o in other_vax:
        df_filt = df_filt[~df_filt['text'].str.lower().str.contains(o)]
    print('vaccine ', vax, len(df_filt))
    return df_filt

covaxin = filtered_vacc(data, ['covaxin'])
sinopharm = filtered_vacc(data, ['sinopharm'])
sinovac = filtered_vacc(data, ['sinovac'])
moderna = filtered_vacc(data, ['moderna'])
pfizer = filtered_vacc(data, ['pfizer','biontech'])
oxford = filtered_vacc(data, ['oxford','astrazeneca'])
sputnik = filtered_vacc(data, ['sputnik'])


# In[69]:


DataExploration.generate_word_cloud(list(moderna['explore_text']))
print(tfidf.get_top_n_tfidf_bow(list(moderna['explore_text']), top_n_tokens = 30))
no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(list(moderna['explore_text']), num_topics = no_topics)
lda.lda_topics(lda_allbow)


# In[70]:


DataExploration.generate_word_cloud(list(pfizer['explore_text']))
print(tfidf.get_top_n_tfidf_bow(list(pfizer['explore_text']), top_n_tokens = 30))
no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(list(pfizer['explore_text']), num_topics = no_topics)
lda.lda_topics(lda_allbow)


# In[71]:


DataExploration.generate_word_cloud(list(oxford['explore_text']))
print(tfidf.get_top_n_tfidf_bow(list(oxford['explore_text']), top_n_tokens = 30))
no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(list(oxford['explore_text']), num_topics = no_topics)
lda.lda_topics(lda_allbow)


# In[72]:


DataExploration.generate_word_cloud(list(sputnik['explore_text']))
print(tfidf.get_top_n_tfidf_bow(list(sputnik['explore_text']), top_n_tokens = 30))
no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(list(sputnik['explore_text']), num_topics = no_topics)
lda.lda_topics(lda_allbow)


# In[73]:


DataExploration.generate_word_cloud(list(sinopharm['explore_text']))
print(tfidf.get_top_n_tfidf_bow(list(sinopharm['explore_text']), top_n_tokens = 30))
no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(list(sinopharm['explore_text']), num_topics = no_topics)
lda.lda_topics(lda_allbow)


# In[74]:


DataExploration.generate_word_cloud(list(sinovac['explore_text']))
print(tfidf.get_top_n_tfidf_bow(list(sinovac['explore_text']), top_n_tokens = 30))
no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(list(sinovac['explore_text']), num_topics = no_topics)
lda.lda_topics(lda_allbow)


# In[75]:


DataExploration.generate_word_cloud(list(covaxin['explore_text']))
print(tfidf.get_top_n_tfidf_bow(list(covaxin['explore_text']), top_n_tokens = 30))
no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(list(covaxin['explore_text']), num_topics = no_topics)
lda.lda_topics(lda_allbow)


# ## Sentiment and Emotion analysis
# with result generated by fastai and fastai with transformer, we have sentiment and emotion classification, we will analyze the dataset with these 2 variable to see how's people's altitude to covid vaccine.
# The code of fastai and fastai with transformer can be found in the same folder, and the code was run in Google colab with Python version 3.7

# In[ ]:




