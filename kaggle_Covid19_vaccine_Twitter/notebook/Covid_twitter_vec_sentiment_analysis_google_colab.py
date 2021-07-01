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


# In[25]:


# covid_tweet = pd.read_csv(os.path.join(path, "vaccination_all_tweets.csv"))
covid_tweet = pd.read_csv(os.path.join(path, "covidvacc_tweet_sentiment.csv"))

covid_tweet = covid_tweet[covid_tweet.text.isnull()==False].drop_duplicates().reset_index(drop=True)
covid_tweet = tweet_proc(covid_tweet,'text')
# covid_tweet['emotion'] = np.nan
covid_tweet['sentiment'] = np.nan
covid_tweet.head(3)


# In[26]:


basic_tweet = pd.read_csv(os.path.join(path, "tweet_dataset.csv"))
basic_tweet = basic_tweet[basic_tweet.sentiment!='empty'].drop_duplicates().reset_index(drop=True)
basic_tweet = basic_tweet[['sentiment','new_sentiment','old_text']].rename(columns={'old_text':'text', 'sentiment':'emotion', 'new_sentiment':'sentiment'})
basic_tweet = tweet_proc(basic_tweet,'text')
basic_tweet.head(3)


# In[27]:


df_lm = basic_tweet[['text', 'emotion','sentiment']].append(covid_tweet[['text', 'emotion','sentiment']]) 
df_clas = df_lm[['text', 'emotion','sentiment']].dropna(subset=['emotion']).dropna(subset=['sentiment'])
print(len(df_lm), len(df_clas))
df_clas.head(3)


# In[29]:


dls_lm = TextDataLoaders.from_df(df_lm, text_col='text', is_lm=True, valid_pct=0.1)
torch.save(dls_lm, os.path.join(path, 'dls_lm.pkl'))


# In[30]:


dls_lm.show_batch(max_n=2)


# In[31]:


learn = language_model_learner(dls_lm, AWD_LSTM, drop_mult = 0.3, metrics=[accuracy, Perplexity()]).to_fp16()


# In[15]:


learn.unfreeze()
learn.fit_one_cycle(4, 1e-3) #4 means 4 epoch
learn.save_encoder(os.path.join(path, 'finetuned_lm')) # save encoder


# In[16]:


TEXT = "I love"
N_WORDS = 45
N_SENTENCES = 2
print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))


# In[6]:


dls_lm= torch.load(os.path.join(path, 'dls_lm.pkl'))


# In[11]:


dls_clas = DataBlock(
    blocks = (TextBlock.from_df('text', seq_len = dls_lm.seq_len, vocab = dls_lm.vocab), CategoryBlock),
    # blocks = (TextBlock.from_df('text', seq_len = 72, vocab = vocab_list), CategoryBlock),
    get_x = ColReader('text'),
    get_y = ColReader('emotion'),
    splitter = RandomSplitter()
).dataloaders(df_clas, bs = 64)
torch.save(dls_clas, os.path.join(path, 'dls_clas_emo.pkl'))


# In[28]:


dls_clas = torch.load(os.path.join(path, 'dls_clas_emo.pkl'))


# In[29]:


dls_clas.show_batch(max_n=4)


# In[13]:


learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy).to_fp16()


# In[14]:


learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3))
learn.save(os.path.join(path, 'twitter_classifier_emo'))


# In[21]:


learn.predict("I am so worried")


# In[22]:


learn.dls.vocab[1]


# Predict

# In[23]:


pred_dl = dls_clas.test_dl(covid_tweet['text'])
preds = learn.get_preds(dl=pred_dl)
# Get predicted sentiment
covid_tweet['emotion'] = preds[0].argmax(dim=-1)


# In[24]:


covid_tweet['emotion'] = covid_tweet['emotion'].map({0:'anger', 1:'boredom', 2:'enthusiasm', 3:'fun', 4:'happiness', 5:'hate', 6:'love', 7:'neutral', 8:'relief', 9:'sadness', 10:'surprise', 11:'worry'})

# Save to csv
covid_tweet.to_csv(os.path.join(path, 'covidvacc_tweet_sentiment.csv'))

# Plot sentiment value counts
covid_tweet['emotion'].value_counts(normalize=True).plot.bar();


# In[32]:


dls_clas = torch.load(os.path.join(path, 'dls_clas_sentiment_cpu.pkl'))
learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy).load(os.path.join(path, 'twitter_classifier_sentiment'))
learn.dls.vocab[1]


# In[33]:


covid_tweet.head(3)


# In[34]:



pred_dl = dls_clas.test_dl(covid_tweet['text'])
preds = learn.get_preds(dl=pred_dl)
# Get predicted sentiment
covid_tweet['sentiment'] = preds[0].argmax(dim=-1)
covid_tweet['sentiment'] = covid_tweet['sentiment'].map({0:'negative', 1:'neutral', 2:'positive'})

# Save to csv
covid_tweet.to_csv(os.path.join(path, 'covidvacc_tweet_sentiment_emo.csv'))

# Plot sentiment value counts
covid_tweet['sentiment'].value_counts(normalize=True).plot.bar();

