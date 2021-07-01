#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import io
import os
import re
from google.colab import drive
get_ipython().system('pip install transformers')
get_ipython().system('pip install SentencePiece')
import torch
import transformers
from transformers import XLNetTokenizer, XLNetModel, AdamW, get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import defaultdict
from textwrap import wrap
from pylab import rcParams

from torch import nn, optim
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset,RandomSampler,SequentialSampler
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from transformers import XLNetTokenizer
from keras.preprocessing.sequence import pad_sequences
drive.mount('/content/drive')


# In[ ]:


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


# In[ ]:


data = pd.read_csv(os.path.join(path, "covid-19_articles_data.csv"))
data = text_proc(data,'text').dropna(subset=['sentiment'])#.sample(3000, random_state = 10).reset_index(drop=True)
data['len'] = data.text.apply(lambda x: len(x.split(' ')))
print(len(data))
data.head(3)


# In[ ]:





# In[ ]:


df_train, df_val = train_test_split(data, test_size=0.33, random_state=42)
len(df_train)


# In[ ]:


df_train.tail(1)


# In[ ]:


# max_len = int(df_train.len.quantile(0.95))
max_len = 300


# In[ ]:


class XLNet_Dataset(Dataset):

    def __init__(self, text, sentiment, tokenizer, max_len):
        self.text = text
        self.sentiment = sentiment
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, item):
        text = str(self.text[item])
        sentiment = self.sentiment[item]

        encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        pad_to_max_length=False,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
        )

        input_ids = pad_sequences(encoding['input_ids'], maxlen=max_len, dtype=torch.Tensor ,truncating="post",padding="post")
        input_ids = input_ids.astype(dtype = 'int64')
        input_ids = torch.tensor(input_ids) 

        attention_mask = pad_sequences(encoding['attention_mask'], maxlen=max_len, dtype=torch.Tensor ,truncating="post",padding="post")
        attention_mask = attention_mask.astype(dtype = 'int64')
        attention_mask = torch.tensor(attention_mask)       

        return {
        'text': text,
        'input_ids': input_ids,
        'attention_mask': attention_mask.flatten(),
        'sentiment': torch.tensor(sentiment, dtype=torch.long)
        }


# In[ ]:


def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = XLNet_Dataset(
    text=df.text.to_numpy(),
    sentiment=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=2,
    drop_last = True
  )


# In[ ]:


BATCH_SIZE = 10
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
train_data_loader = create_data_loader(df_train, tokenizer, max_len, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, max_len, BATCH_SIZE)


# In[ ]:


device = torch.device('cuda:0')
# device = torch.device('cpu')


# In[ ]:


from transformers import XLNetForSequenceClassification
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels = 2)
model = model.to(device)


# In[ ]:


EPOCHS = 2
BATCH_SIZE = 10

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
                                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)


# In[ ]:


from sklearn import metrics
def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples, batch_size, maxlen):
    model = model.train()
    losses = []
    acc = 0
    counter = 0
  
    for d in data_loader:
        input_ids = d["input_ids"].reshape(batch_size,maxlen).to(device)
        attention_mask = d["attention_mask"].to(device)
        sentiment = d["sentiment"].to(device)
        
        outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels = sentiment)
        loss = outputs[0]
        logits = outputs[1]

        # preds = preds.cpu().detach().numpy()
        _, prediction = torch.max(outputs[1], dim=1)
        sentiment = sentiment.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(sentiment, prediction)

        acc += accuracy
        losses.append(loss.item())
        
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        counter = counter + 1

    return acc / counter, np.mean(losses)


# In[ ]:


def eval_model(model, data_loader, device, n_examples,batch_size,maxlen):
    model = model.eval()
    losses = []
    acc = 0
    counter = 0
  
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].reshape(batch_size,maxlen).to(device)
            attention_mask = d["attention_mask"].to(device)
            sentiment = d["sentiment"].to(device)
            
            outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels = sentiment)
            loss = outputs[0]
            logits = outputs[1]

            _, prediction = torch.max(outputs[1], dim=1)
            sentiment = sentiment.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            accuracy = metrics.accuracy_score(sentiment, prediction)

            acc += accuracy
            losses.append(loss.item())
            counter += 1

    return acc / counter, np.mean(losses)


# In[145]:


get_ipython().run_cell_magic('time', '', "history = defaultdict(list)\nbest_accuracy = 0\n\nfor epoch in range(EPOCHS):\n    print(f'Epoch {epoch + 1}/{EPOCHS}')\n    print('-' * 10)\n\n    train_acc, train_loss = train_epoch(\n        model,\n        train_data_loader,     \n        optimizer, \n        device, \n        scheduler, \n        len(df_train),\n        BATCH_SIZE,\n        max_len\n    )\n\n    print(f'Train loss {train_loss} Train accuracy {train_acc}')\n\n    val_acc, val_loss = eval_model(\n        model,\n        val_data_loader, \n        device, \n        len(df_val),\n        BATCH_SIZE,\n        max_len\n    )\n\n    print(f'Val loss {val_loss} Val accuracy {val_acc}')\n    print()\n\n    history['train_acc'].append(train_acc)\n    history['train_loss'].append(train_loss)\n    history['val_acc'].append(val_acc)\n    history['val_loss'].append(val_loss)\n\n    if val_acc > best_accuracy:\n        torch.save(model.state_dict(), os.path.join(path, 'xlnet_model.bin'))\n        best_accuracy = val_acc")


# In[ ]:


def predict_sentiment(text):
    review_text = text

    encoded_review = tokenizer.encode_plus(
    review_text,
    max_length=MAX_LEN,
    add_special_tokens=True,
    return_token_type_ids=False,
    pad_to_max_length=False,
    return_attention_mask=True,
    return_tensors='pt',
    )

    input_ids = pad_sequences(encoded_review['input_ids'], maxlen=MAX_LEN, dtype=torch.Tensor ,truncating="post",padding="post")
    input_ids = input_ids.astype(dtype = 'int64')
    input_ids = torch.tensor(input_ids) 

    attention_mask = pad_sequences(encoded_review['attention_mask'], maxlen=MAX_LEN, dtype=torch.Tensor ,truncating="post",padding="post")
    attention_mask = attention_mask.astype(dtype = 'int64')
    attention_mask = torch.tensor(attention_mask) 

    input_ids = input_ids.reshape(1,512).to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    outputs = outputs[0][0].cpu().detach()

    probs = F.softmax(outputs, dim=-1).cpu().detach().numpy().tolist()
    _, prediction = torch.max(outputs, dim =-1)

    print("Positive score:", probs[1])
    print("Negative score:", probs[0])
    print(f'Review text: {review_text}')
    print(f'Sentiment  : {class_names[prediction]}')

