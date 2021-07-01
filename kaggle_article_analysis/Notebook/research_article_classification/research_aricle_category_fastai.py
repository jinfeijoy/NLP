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
get_ipython().system('pip install fastai==2.3.1')
from fastai.text.all import *


# In[3]:


def generate_target(data):
    tmp = data.copy()
    tmp.set_index(['ID','TITLE','ABSTRACT'],inplace=True)
    tmp['LABEL'] = tmp.idxmax(axis=1)
    tmp = tmp.reset_index()
    tmp = tmp[['ID','TITLE','ABSTRACT','LABEL']]
    return tmp
path = '/content/drive/MyDrive/colab_data'
data = pd.read_csv(os.path.join(path, "train.csv"))
data = generate_target(data)
convert_label = {'Computer Science':0,'Physics':1,'Mathematics':2,'Statistics':3,'Quantitative Biology':4,'Quantitative Finance':5}
data['convert_label'] = data.LABEL.map(convert_label)


# In[4]:


df_clas = data[['TITLE', 'LABEL']].dropna(subset=['LABEL'])
df_clas.columns = df_clas.columns.str.strip()


# In[6]:


dls_lm = TextDataLoaders.from_df(data, text_col='TITLE', is_lm=True, valid_pct=0.1)
dls_lm.show_batch(max_n=2)
# torch.save(dls_lm, os.path.join(path, 'dls_lm.pkl'))


# In[7]:


learn = language_model_learner(dls_lm, AWD_LSTM, drop_mult = 0.3, metrics=[accuracy, Perplexity()]).to_fp16()


# In[8]:


learn.unfreeze()
learn.fit_one_cycle(4, 1e-3) 


# In[9]:


TEXT = "Apple"
N_WORDS = 45
N_SENTENCES = 2
print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))


# In[11]:


from sklearn.model_selection import train_test_split
df_train, df_val = train_test_split(df_clas, test_size=0.33, random_state=42)


# In[13]:


dls_clas = DataBlock(
    blocks = (TextBlock.from_df('TITLE', seq_len = dls_lm.seq_len, vocab = dls_lm.vocab), CategoryBlock),
    get_x = ColReader('text'),
    get_y = ColReader('LABEL'),
    splitter = RandomSplitter()
).dataloaders(df_train, bs = 64)


# In[16]:


learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy).to_fp16()


# In[17]:


learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-3/(2.6**4),1e-3))


# In[18]:


pred_dl = dls_clas.test_dl(df_val['TITLE'])
preds = learn.get_preds(dl=pred_dl)


# In[25]:


learn.dls.vocab[1]


# In[19]:


preds


# In[26]:


# convert_label = {'Computer Science':0,'Physics':1,'Mathematics':2,'Statistics':3,'Quantitative Biology':4,'Quantitative Finance':5}
# data['convert_label'] = data.LABEL.map(convert_label)
cols = ['Computer Science', 'Mathematics', 'Physics', 'Quantitative Biology', 'Quantitative Finance', 'Statistics']
predsTest = pd.DataFrame(np.asarray(preds[0]), columns = cols)
predsTest['pred'] = predsTest.idxmax(axis = 1)
# predsTest['pred_convert'] = predsTest['pred'].apply(lambda x: target_map[x])


# In[27]:


from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(predsTest.pred, df_val.LABEL)


# In[28]:


print(classification_report(predsTest.pred, df_val.LABEL))

