#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Please select GPU first (from Edit->NotebookSetting)
import pandas as pd
import numpy as np
import io
import os
import re
from google.colab import drive
get_ipython().system('pip install fastai==2.3.1')
from fastai.text.all import *
drive.mount('/content/drive')


# In[2]:


path = '/content/drive/MyDrive/colab_data'
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


# In[3]:


covid_tweet = pd.read_csv(os.path.join(path, "Covid-19 Twitter Dataset (Aug-Sep 2020).csv"))
covid_tweet = covid_tweet[covid_tweet.original_text.isnull()==False].drop_duplicates().reset_index(drop=True)
covid_tweet = tweet_proc(covid_tweet,'original_text')
covid_tweet['emotion'] = np.nan
covid_tweet = covid_tweet[covid_tweet.lang=='en']
covid_tweet = covid_tweet.rename(columns={'original_text':'text'})
# covid_tweet = covid_tweet.sample(n=40000, random_state=1)
covid_tweet.head(3)


# In[4]:


basic_tweet = pd.read_csv(os.path.join(path, "tweet_dataset.csv"))
basic_tweet = basic_tweet[basic_tweet.sentiment!='empty'].drop_duplicates().reset_index(drop=True)
basic_tweet = basic_tweet[['sentiment','old_text']].rename(columns={'old_text':'text', 'sentiment':'emotion'})
basic_tweet = tweet_proc(basic_tweet,'text')
basic_tweet.head(3)


# In[5]:


df_lm = basic_tweet[['text', 'emotion']].append(covid_tweet[['text', 'emotion']]) 
df_clas = df_lm[['text', 'emotion']].dropna(subset=['emotion'])
print(len(df_lm), len(df_clas))
df_clas.head(3)


# In[6]:


dls_lm= torch.load(os.path.join(path, 'dls_lm.pkl'))


# In[8]:


dls_clas = DataBlock(
    blocks = (TextBlock.from_df('text', seq_len = dls_lm.seq_len, vocab = dls_lm.vocab), CategoryBlock),
    # blocks = (TextBlock.from_df('text', seq_len = 72, vocab = vocab_list), CategoryBlock),
    get_x = ColReader('text'),
    get_y = ColReader('emotion'),
    splitter = RandomSplitter()
).dataloaders(df_clas, bs = 64)
torch.save(dls_clas, os.path.join(path, 'dls_clas_emo.pkl'))


# In[7]:


# load file
dls_clas = torch.load(os.path.join(path, 'dls_clas_emo.pkl'))


# In[8]:


dls_clas.show_batch(max_n=4)


# In[10]:


learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy).to_fp16()


# In[12]:


learn.unfreeze()
learn.fit_one_cycle(4, slice(1e-3/(2.6**4),1e-3))
learn.save(os.path.join(path, 'twitter_classifier_emo'))


# In[ ]:


learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy).load(os.path.join(path, 'twitter_classifier_emo'))


# In[10]:


learn.predict("I hate")


# In[11]:


basic_tweet.emotion.value_counts()


# In[12]:


learn.dls.vocab[1]


# In[13]:


pred_dl = dls_clas.test_dl(covid_tweet['text'])
preds = learn.get_preds(dl=pred_dl)
# Get predicted sentiment
covid_tweet['emo'] = preds[0].argmax(dim=-1)
covid_tweet['emo'] = covid_tweet['emo'].map({0:'anger', 1:'boredom', 2:'enthusiasm', 3:'fun', 4:'happiness', 5:'hate', 6:'love', 7:'neutral', 8:'relief', 9:'sadness', 10:'surprise', 11:'worry'})

# Save to csv
covid_tweet.to_csv(os.path.join(path, 'covid_tweet_sentiment_emo.csv'))

# Plot sentiment value counts
covid_tweet['emo'].value_counts(normalize=True).plot.bar();


# In[ ]:




