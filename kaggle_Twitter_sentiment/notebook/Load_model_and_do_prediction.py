#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Please select GPU first (from Edit->NotebookSetting)
import pandas as pd
import numpy as np
import io
import os
import re
get_ipython().system('pip install fastai==2.3.1')
from fastai.text.all import *
from google.colab import drive
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
covid_tweet['label'] = np.nan
covid_tweet = covid_tweet[covid_tweet.lang=='en']
covid_tweet = covid_tweet.rename(columns={'original_text':'text'})
# covid_tweet = covid_tweet.sample(n=40000, random_state=1)
covid_tweet.head(3)


# In[4]:


basic_tweet = pd.read_csv(os.path.join(path, "tweet_dataset.csv"))
basic_tweet = basic_tweet[basic_tweet.new_sentiment.isnull()==False].drop_duplicates().reset_index(drop=True)
basic_tweet = basic_tweet[['old_text', 'new_sentiment']].rename(columns={'old_text':'text', 'new_sentiment':'label'})
basic_tweet = tweet_proc(basic_tweet,'text')
# basic_tweet = basic_tweet[['id', 'text', 'label']]
basic_tweet.head(3)


# # Predict next word

# In[5]:


dls_lm= torch.load(os.path.join(path, 'dls_lm.pkl'))


# In[10]:


learn = language_model_learner(dls_lm, AWD_LSTM, drop_mult = 0.3, metrics=[accuracy, Perplexity()]).to_fp16()


# In[12]:


learn.unfreeze()
learn.fit_one_cycle(4, 1e-3)


# In[18]:


learn.save(os.path.join(path, 'finetuned_lm_0531'))


# In[20]:


learn = language_model_learner(dls_lm, AWD_LSTM, drop_mult = 0.3, metrics=[accuracy, Perplexity()]).load_encoder(os.path.join(path, 'finetuned_lm'))


# be careful here, `lean.save` -> `load`, `lean.save_encoder` -> `load_encoder`

# In[23]:


TEXT = "Today"
N_WORDS = 60
N_SENTENCES = 2
print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))


# # Classification

# In[7]:


dls_clas = torch.load(os.path.join(path, 'dls_clas.pkl'))


# In[8]:


learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy).load(os.path.join(path, 'twitter_classifier'))


# In[9]:


TEXT = 'I like this movie'
learn.predict(TEXT)


# In[ ]:


pred_dl = dls_clas.test_dl(covid_tweet['text'])
preds = learn.get_preds(dl=pred_dl)


# In[ ]:


covid_tweet['label'] = preds[0].argmax(dim=-1)
covid_tweet['label'] = covid_tweet['label'].map({0:'negative', 1:'positive'})

# Save to csv
covid_tweet.to_csv(os.path.join(path, 'covid_tweet_sentiment.csv'))

