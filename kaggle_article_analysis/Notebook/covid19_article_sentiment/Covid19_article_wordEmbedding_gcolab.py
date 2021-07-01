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
drive.mount('/content/drive')
import torch
import torch.optim as optim
import random 
get_ipython().system('pip install fastai==2.3.1')
from fastai.text.all import *


# In[2]:


path = '/content/drive/MyDrive/colab_data'
def de_emojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')
def text_proc(df, text_col='text'):
    df['orig_text'] = df[text_col]
    # Remove twitter handles
    df[text_col] = df[text_col].apply(lambda x:re.sub('@[^\s]+','',x))
    # Remove URLs
    df[text_col] = df[text_col].apply(lambda x:x.replace('<br />', ' '))
    return df[df[text_col]!='']


# In[3]:


data = pd.read_csv(os.path.join(path, "covid-19_articles_data.csv"))
# data = data[data.sentiment!='empty'].drop_duplicates().sample(1000, random_state = 10).reset_index(drop=True)
data = text_proc(data,'text').dropna(subset=['sentiment'])
print(len(data))
data.head(3)


# # AWD-LSTM

# In[4]:


dls_lm = TextDataLoaders.from_df(data, text_col='text', is_lm=True, valid_pct=0.1)


# In[5]:


learn = language_model_learner(dls_lm, AWD_LSTM, drop_mult = 0.3, metrics=[accuracy, Perplexity()]).to_fp16()


# In[6]:


learn.unfreeze()
learn.fit_one_cycle(4, 1e-3) #4 means 4 epoch


# In[13]:


learn.predict("If you've recently heard from an old friend, you're not alone. ", 100, temperature=0.75)


# In[8]:


dls_clas = DataBlock(
    blocks = (TextBlock.from_df('text', seq_len = dls_lm.seq_len, vocab = dls_lm.vocab), CategoryBlock),
    # blocks = (TextBlock.from_df('text', seq_len = 72, vocab = vocab_list), CategoryBlock),
    get_x = ColReader('text'),
    get_y = ColReader('sentiment'),
    splitter = RandomSplitter()
).dataloaders(data, bs = 64)


# In[15]:


classlearn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy).to_fp16()
classlearn.unfreeze()
classlearn.fit_one_cycle(5, slice(1e-3/(2.6**4),1e-3))


# In[17]:


pred_dl = dls_clas.test_dl(data['text'])
preds = classlearn.get_preds(dl=pred_dl)


# 
