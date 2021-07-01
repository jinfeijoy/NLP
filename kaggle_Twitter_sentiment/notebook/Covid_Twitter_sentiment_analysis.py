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


# In[2]:


raw_data = pd.read_csv(os.path.join(root_path, "Covid-19 Twitter Dataset (Aug-Sep 2020).csv"))


# In[4]:


raw_data.head(3)


# In[3]:


data = raw_data[raw_data.original_text.isnull()==False]
data = data.drop_duplicates()
len(data)


# In[6]:


data.head(3)


# If there is no clean_tweet, use the textClean.pipeline to preprocess the data. Use textClean.get_hashtag to generate hashtag.

# In[124]:


# data['clean_tweet'] = textClean.pipeline(data['original_text'].to_list(), multi_gram = [1], 
#                                        remove_pattern = ['http','@','#'],
#                                        lower_case=True, deacc=False, encoding='utf8',
#                                        errors='strict', stem_lemma = 'stem', tag_drop = [''], 
#                                        nltk_stop=True, stop_word_list=['rt'], 
#                                        check_numbers=True, word_length=2, remove_consecutives=True)


# In[16]:


# preprocessed_text = textClean.pipeline(data['original_text'][0:40].to_list(), multi_gram = [1], 
#                                        remove_pattern = ['http','@','#'],
#                                        lower_case=True, deacc=False, encoding='utf8',
#                                        errors='strict', stem_lemma = 'stem', tag_drop = [''], 
#                                        nltk_stop=True, stop_word_list=['rt'], 
#                                        check_numbers=True, word_length=2, remove_consecutives=True)
# preprocessed_text = [' '.join(i) for i in preprocessed_text]
# data['hashtag'] = data.clean_tweet.apply(lambda x: textClean.get_hashtag(x)) 


# In[8]:


i = 3
print(data.original_text[i])
print("-----------")
print(data.clean_tweet[i])


# ## Vader
# Apply vader and generate neg, neu, pos and compound. This dataset already have these columns, this analysis can be done in the future task. Reference: https://predictivehacks.com/how-to-run-sentiment-analysis-in-python-using-vader/

# In[93]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import twitter_samples


# In[97]:


vader_analyzer = SentimentIntensityAnalyzer()
vader_analyzer.polarity_scores(data.clean_tweet[0])
# data['neg'] = data['clean_tweet'].apply(lambda x: vader_analyzer.polarity_scores(x)['neg'])
# data['neu'] = data['clean_tweet'].apply(lambda x: vader_analyzer.polarity_scores(x)['neu'])
# data['pos'] = data['clean_tweet'].apply(lambda x: vader_analyzer.polarity_scores(x)['pos'])
# data['compound'] = data['clean_tweet'].apply(lambda x: vader_analyzer.polarity_scores(x)['compound'])


# ## Explore Data

# In[ ]:





# In[5]:


pos_tweet = list(x.split() for x in data[data['sentiment']=='pos']['clean_tweet'])
neg_tweet = list(x.split() for x in data[data['sentiment']=='neg']['clean_tweet'])
# neu_tweet = list(x.split() for x in data[data['sentiment']=='neu']['clean_tweet'])


# In[35]:


postop10tfidf = tfidf.get_top_n_tfidf_bow(pos_tweet, top_n_tokens = 30)
negtop10tfidf = tfidf.get_top_n_tfidf_bow(neg_tweet, top_n_tokens = 30)
# neutop10tfidf = tfidf.get_top_n_tfidf_bow(neu_tweet, top_n_tokens = 30)
print('top 30 negative review tfidf', negtop10tfidf)
print('top 30 positive review tfidf', postop10tfidf)
# print('top 30 neutual review tfidf', neutop10tfidf)


# In[36]:


top10_posfreq_list = DataExploration.get_topn_freq_bow(pos_tweet, topn = 10)
top10_negfreq_list = DataExploration.get_topn_freq_bow(neg_tweet, topn = 10)
print(top10_posfreq_list)
print(top10_negfreq_list)


# In[6]:


DataExploration.generate_word_cloud(pos_tweet)


# In[7]:


DataExploration.generate_word_cloud(neg_tweet)


# ## LDA

# In[45]:


no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(pos_tweet, num_topics = no_topics)
# lda_top30bow, bow_corpus, dictionary  = lda.fit_lda(pos_tweet, top_n_tokens = 30, num_topics = no_topics)
lda.lda_topics(lda_allbow)


# In[46]:


no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(neg_tweet, num_topics = no_topics)
# lda_top30bow, bow_corpus, dictionary  = lda.fit_lda(neg_tweet, top_n_tokens = 30, num_topics = no_topics)
lda.lda_topics(lda_allbow)

