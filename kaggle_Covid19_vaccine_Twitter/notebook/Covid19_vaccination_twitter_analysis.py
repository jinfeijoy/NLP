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


# In[3]:


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
def get_clean_country(text):
    for country in pycountry.countries:
        if country.name in text:
            return country.name
def get_vacc_brand(sentence):
    vax_dict = {
    'covaxin':'covaxin', 
    'sinopharm':'sinopharm', 
    'sinovac':'sinovac', 
    'moderna':'moderna', 
    'pfizer':'pfizer', 
    'biontech':'pfizer', 
    'oxford':'astrazeneca', 
    'astrazeneca':'astrazeneca', 
    'sputnik':'sputnik'
    }
    for o in list(vax_dict.keys()):
        if sentence.lower().__contains__(o):
            return vax_dict.get(o)
    else:
        return 'Others'


# In[4]:


raw_data = pd.read_csv(os.path.join(root_path, "covidvacc_tweet_sentiment_emo.csv"))
data = raw_data[raw_data.text.isnull()==False]
# data = data[['user_name','date','user_location','user_description','user_created','user_followers','user_friends','user_favourites','user_verified','text','hashtags','source','orig_text','emotion','sentiment']]
data = data.drop_duplicates()
data = tweet_proc(data,'text')
data['country'] = data.user_location.apply(lambda x: get_clean_country(str(x)))
data['date']=pd.to_datetime(data['date'], errors='coerce').dt.date
data['vacc_brand'] = data.text.apply(lambda x: get_vacc_brand(x))
# model accuracy from fastai with twitter pre-trained model: sentiment: 0.70, emotion: 0.35


# In[4]:


data.head(3)


# In[5]:


len(data)


# In[85]:


len(raw_data)


# ## Analyze Sentiment over Time for different country

# In[9]:


sentiment_overtime = data.groupby(['date', 'sentiment']).agg(**{'tweets': ('id', 'count')}).reset_index().dropna()
fig = px.line(sentiment_overtime, x='date', y='tweets', color='sentiment', category_orders={'sentiment': ['neutral', 'negative', 'positive']},
             title='Timeline showing sentiment of tweets about COVID-19 vaccines')
fig.show()


# From the above chart we can see, there is a spike in March 1st 2021, and later in April, to analyze the spike, we need to more exploration.

# In[15]:


emo_overtime = data[~data.emotion.isin(['neutral','worry'])].groupby(['date', 'emotion']).agg(**{'tweets': ('id', 'count')}).reset_index().dropna()
fig = px.line(emo_overtime, x='date', y='tweets', color='emotion', category_orders={'emotion': ['anger', 'boredom', 'enthusiasm', 'fun', 'happiness', 'hate', 'love', 'neutral', 'relief', 'sadness', 'surprise', 'worry']},
             title='Timeline showing emotion of tweets about COVID-19 vaccines')
fig.show()


# In[30]:


country_overtime = data[~data.country.isin(['India'])].groupby(['date', 'country']).agg(**{'tweets': ('id', 'count')}).reset_index().dropna()
fig = px.line(country_overtime, x='date', y='tweets', color='country',
             title='Timeline showing country of tweets about COVID-19 vaccines')
fig.show()


# In[75]:


india_overtime = data[data.country.isin(['India'])].groupby(['date', 'sentiment']).agg(**{'tweets': ('id', 'count')}).reset_index().dropna()
fig = px.line(india_overtime, x='date', y='tweets', color='sentiment',
             title='Timeline showing emotion of tweets in India about COVID-19 vaccines')
fig.show()


# In[48]:


test= data[(data.date.astype(str)=='2021-06-01')&(data.country=='India')].reset_index(drop=True)
test.orig_text[5]


# In[49]:


test['explore_text'] = textClean.pipeline(test['text'].to_list(), multi_gram = [1], lower_case=True, 
                                                 deacc=False, encoding='utf8', errors='strict', stem_lemma = 'lemma', 
                                                 tag_drop = [], nltk_stop=True, 
                                                 stop_word_list=['effect','vaccine','side','covid'], 
                                                 check_numbers=False, word_length=2, remove_consecutives=True)
print(tfidf.get_top_n_tfidf_bow(list(test['explore_text']), top_n_tokens = 30))
DataExploration.generate_word_cloud(list(test['explore_text']))
no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(list(test['explore_text']), num_topics = 10)
lda.lda_topics(lda_allbow)


# From the above plots we can see, people in India and Canada tweets a lot in March and April, and the spikes in overall volume trend were contributed by India, we will do more exploration for Canada and India. For india, it seems most tweets are neutral.
# 
# In India, in 2021-Mar-01, most tweets are about vacc, in 2021-Apr-21, most tweets are about vacc and infection and medical service, in 2021-June-01, India start use sputnik, etc.

# In[76]:


canada_overtime = data[data.country.isin(['Canada'])].groupby(['date', 'sentiment']).agg(**{'tweets': ('id', 'count')}).reset_index().dropna()
fig = px.line(canada_overtime, x='date', y='tweets', color='sentiment',
             title='Timeline showing emotion of tweets in Canada about COVID-19 vaccines')
fig.show()


# In[10]:


canada_overtime = data[(data.country.isin(['Canada']))&(data.sentiment == 'negative')].groupby(['date', 'emotion']).agg(**{'tweets': ('id', 'count')}).reset_index().dropna()
fig = px.line(canada_overtime, x='date', y='tweets', color='emotion',
             title='Timeline showing emotion of tweets in Canada about COVID-19 vaccines')
fig.show()


# In[53]:


test= data[(data.date.astype(str)=='2021-05-25')&(data.country=='Canada')].reset_index(drop=True)
test.orig_text[20]
test['explore_text'] = textClean.pipeline(test['text'].to_list(), multi_gram = [1], lower_case=True, 
                                                 deacc=False, encoding='utf8', errors='strict', stem_lemma = 'lemma', 
                                                 tag_drop = [], nltk_stop=True, 
                                                 stop_word_list=['effect','vaccine','side','covid'], 
                                                 check_numbers=False, word_length=2, remove_consecutives=True)
print(tfidf.get_top_n_tfidf_bow(list(test['explore_text']), top_n_tokens = 30))
DataExploration.generate_word_cloud(list(test['explore_text']))
no_topics = 4
lda_allbow, bow_corpus, dictionary = lda.fit_lda(list(test['explore_text']), num_topics = 4)
lda.lda_topics(lda_allbow)


# For Canada, most tweets are neutral and negative, negative mostly are worry, especially after 2021 April. After check tweets in Apr-13, it seems the negative sentiment come from: bad news (death) from oxford vacc, shortgage of vacc, vacc side effect, low fully vaccinated rate in Canada.

# In[37]:


china_overtime = data[data.country.isin(['China'])].groupby(['date', 'emotion']).agg(**{'tweets': ('id', 'count')}).reset_index().dropna()
fig = px.line(china_overtime, x='date', y='tweets', color='emotion', category_orders={'emotion': ['anger', 'boredom', 'enthusiasm', 'fun', 'happiness', 'hate', 'love', 'neutral', 'relief', 'sadness', 'surprise', 'worry']},
             title='Timeline showing emotion of tweets in China about COVID-19 vaccines')
fig.show()


# In[55]:


test= data[(data.country=='China')].reset_index(drop=True)
print(test.orig_text[20])
test['explore_text'] = textClean.pipeline(test['text'].to_list(), multi_gram = [1], lower_case=True, 
                                                 deacc=False, encoding='utf8', errors='strict', stem_lemma = 'lemma', 
                                                 tag_drop = [], nltk_stop=True, 
                                                 stop_word_list=['effect','vaccine','side','covid'], 
                                                 check_numbers=False, word_length=2, remove_consecutives=True)
print(tfidf.get_top_n_tfidf_bow(list(test['explore_text']), top_n_tokens = 30))
DataExploration.generate_word_cloud(list(test['explore_text']))
no_topics = 4
lda_allbow, bow_corpus, dictionary = lda.fit_lda(list(test['explore_text']), num_topics = 4)
lda.lda_topics(lda_allbow)


# In[8]:


test= data[(data.country=='Pakistan')].reset_index(drop=True)
print(test.orig_text[20])
test['explore_text'] = textClean.pipeline(test['text'].to_list(), multi_gram = [1], lower_case=True, 
                                                 deacc=False, encoding='utf8', errors='strict', stem_lemma = 'lemma', 
                                                 tag_drop = [], nltk_stop=True, 
                                                 stop_word_list=['effect','vaccine','side','covid'], 
                                                 check_numbers=False, word_length=2, remove_consecutives=True)
print(tfidf.get_top_n_tfidf_bow(list(test['explore_text']), top_n_tokens = 30))
DataExploration.generate_word_cloud(list(test['explore_text']))
no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(list(test['explore_text']), num_topics = no_topics)
lda.lda_topics(lda_allbow)


# ## Sentiment analysis for different vacc

# In[88]:


all_vax = ['covaxin', 'sinopharm', 'sinovac', 'moderna', 'pfizer', 'biontech', 'oxford', 'astrazeneca', 'sputnik']

# Note: a lot of the tweets seem to contain hashtags for multiple vaccines even though they are specifically referring to one vaccine - not very helpful!
def filtered_vacc(df, vax):
    df = df.dropna()
    df_filt = pd.DataFrame()
    for o in vax:
        df_filt = df_filt.append(df[df['orig_text'].str.lower().str.contains(o)])
    other_vax = list(set(all_vax)-set(vax))
    for o in other_vax:
        df_filt = df_filt[~df_filt['orig_text'].str.lower().str.contains(o)]
    print('vaccine ', vax, len(df_filt))
    return df_filt

covaxin = filtered_vacc(raw_data, ['covaxin'])
sinopharm = filtered_vacc(raw_data, ['sinopharm'])
sinovac = filtered_vacc(raw_data, ['sinovac'])
moderna = filtered_vacc(raw_data, ['moderna'])
pfizer = filtered_vacc(raw_data, ['pfizer','biontech'])
oxford = filtered_vacc(raw_data, ['oxford','astrazeneca'])
sputnik = filtered_vacc(raw_data, ['sputnik'])


# In[6]:


data_overtime = data.groupby(['vacc_brand', 'sentiment']).agg(**{'tweets': ('id', 'count')}).reset_index().dropna()

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_theme(style="whitegrid")
ax = sns.barplot(x="vacc_brand", y="tweets", hue="sentiment", data=data_overtime[data_overtime.vacc_brand!='Others'])


# ## Side Effect

# In[171]:


def find_side_effect(df):
    df = df.dropna()
    df_filt = pd.DataFrame()
    effect = ['side','effect','side effect','fever','headache']
    for o in effect:
        df_filt = df_filt.append(df[df['orig_text'].str.lower().str.contains('side effect')])
    print('vaccine ', len(df_filt))
    return df_filt

side_effect = find_side_effect(data).reset_index(drop=True)
side_effect = side_effect[side_effect.sentiment=='negative']
side_effect['explore_text'] = textClean.pipeline(side_effect['text'].to_list(), multi_gram = [1], lower_case=True, 
                                                 deacc=False, encoding='utf8', errors='strict', stem_lemma = 'lemma', 
                                                 tag_drop = [], nltk_stop=True, 
                                                 stop_word_list=['effect','vaccine','side','covid'], 
                                                 check_numbers=False, word_length=2, remove_consecutives=True)


# In[172]:


print(DataExploration.get_topn_freq_bow(list(side_effect['explore_text']), topn = 30))


# In[173]:


print(tfidf.get_top_n_tfidf_bow(list(side_effect['explore_text']), top_n_tokens = 30))


# In[174]:


DataExploration.generate_word_cloud(list(side_effect['explore_text']))


# In[162]:


no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(list(side_effect['explore_text']), num_topics = 10)
lda.lda_topics(lda_allbow)


# In[150]:


side_effect.emotion.value_counts()


# In[175]:


def plot_count(feature, title, df, size=1, ordered=True):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    if ordered:
        g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')
    else:
        g = sns.countplot(df[feature], palette='Set3')
    g.set_title("Number and percentage of {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()


# In[186]:


fig = px.histogram(data, x="country")
fig.show()


# In[4]:


dbplot.plot_count_dist("country", "country", data, 4, True)


# In[4]:


dbplot.generate_bar_proportion(data[data.sentiment!='neutral'], 'emotion', 'country', color = 0, order = True, topn = 10)


# In[5]:


dbplot.generate_bar_proportion(data[data.sentiment!='neutral'], 'emotion', 'country', color = 0, order = True, topn = 10)


# In[9]:


dbplot.generate_bar_proportion(data[data.sentiment!='neutral'], 'sentiment', 'date', color = 0, order = False, topn = 10)


# In[6]:


dbplot.generate_bar_proportion(data[data.sentiment!='neutral'], 'sentiment', 'date', color = 0, order = False, topn = 10)


# In[6]:


dbplot.generate_bar_proportion(data[data.sentiment!='neutral'], 'sentiment', 'country', color = 0, order = True, topn = 10)

