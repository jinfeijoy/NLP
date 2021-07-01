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


# In[2]:


path = '/content/drive/MyDrive/colab_data'
def de_emojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')
def clean_text(text):
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ',  text)
    text = re.sub(r" +", ' ', text)
    return text
def text_proc(df, text_col='text'):
    df['orig_text'] = df[text_col]
    # Remove twitter handles
    df[text_col] = df[text_col].apply(lambda x: clean_text(x))
    # Remove URLs
    df[text_col] = df[text_col].apply(lambda x:x.replace('<br />', ' '))
    return df[df[text_col]!='']
data = pd.read_csv(os.path.join(path, "nasdaq.csv"))
data = text_proc(data,'Headline').dropna(subset=['Label'])#.sample(2000, random_state = 10).reset_index(drop=True)
data['len'] = data.Headline.apply(lambda x: len(x.split(' ')))
print(len(data))
print(data['len'].quantile(0.99))
data.head(3)


# In[3]:


df_clas = data[['Headline', 'Label']].dropna(subset=['Label'])
df_clas.columns = df_clas.columns.str.strip()


# In[4]:


dls_lm = TextDataLoaders.from_df(data, text_col='Headline', is_lm=True, valid_pct=0.1)
dls_lm.show_batch(max_n=2)
torch.save(dls_lm, os.path.join(path, 'dls_lm.pkl'))


# In[5]:


learn = language_model_learner(dls_lm, AWD_LSTM, drop_mult = 0.3, metrics=[accuracy, Perplexity()]).to_fp16()


# In[6]:


learn.unfreeze()
learn.fit_one_cycle(4, 1e-3) 


# In[9]:


TEXT = "Apple"
N_WORDS = 45
N_SENTENCES = 2
print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))


# In[18]:


dls_clas = DataBlock(
    blocks = (TextBlock.from_df('Headline', seq_len = dls_lm.seq_len, vocab = dls_lm.vocab), CategoryBlock),
    get_x = ColReader('text'),
    get_y = ColReader('Label'),
    splitter = RandomSplitter()
).dataloaders(df_train, bs = 64)


# In[10]:


dls_clas.show_batch(max_n=4)


# In[42]:


learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy).to_fp16()


# In[43]:


learn.unfreeze()
learn.fit_one_cycle(10, slice(1e-3/(2.6**4),1e-3))


# In[21]:


from sklearn.model_selection import train_test_split
df_train, df_val = train_test_split(df_clas, test_size=0.33, random_state=42)


# In[44]:


pred_dl = dls_clas.test_dl(df_val['Headline'])
preds = learn.get_preds(dl=pred_dl)


# In[45]:


target_map = {'neg':0,'pos':1,'neu':2}
predsTest = pd.DataFrame(np.asarray(preds[0]), columns = ['neg','pos','neu'])
predsTest['pred'] = predsTest.idxmax(axis = 1)
predsTest['pred_convert'] = predsTest['pred'].apply(lambda x: target_map[x])


# In[46]:


from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(predsTest.pred_convert, df_val.Label)


# In[47]:


print(classification_report(predsTest.pred_convert, df_val.Label))

