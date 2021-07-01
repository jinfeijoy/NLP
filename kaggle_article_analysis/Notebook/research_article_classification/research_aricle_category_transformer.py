#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from tqdm.notebook import tqdm
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset,RandomSampler,SequentialSampler
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from transformers import (AutoConfig, 
                          AutoModelForSequenceClassification, 
                          AutoTokenizer, AdamW, 
                          get_linear_schedule_with_warmup,
                          set_seed,
                          )
from keras.preprocessing.sequence import pad_sequences
drive.mount('/content/drive')


# In[2]:


set_seed(123)
epochs = 4
batches = 5
max_len = 145
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'bert-base-cased'
# model_name = 'xlnet-base-cased'
# model_name = 'xlm-roberta-large'
model_name = 'bert-base-uncased'
n_labels = 6
class_names = ['Computer Science','Physics','Mathematics','Statistics','Quantitative Biology','Quantitative Finance']
output_model = 'research_cat.bin'


# In[3]:


def generate_target(data):
    tmp = data.copy()
    tmp.set_index(['ID','TITLE','ABSTRACT'],inplace=True)
    tmp['LABEL'] = tmp.idxmax(axis=1)
    tmp = tmp.reset_index()
    tmp = tmp[['ID','TITLE','ABSTRACT','LABEL']]
    return tmp


# In[5]:


path = '/content/drive/MyDrive/colab_data'
train = pd.read_csv(os.path.join(path, "train.csv"))
test = pd.read_csv(os.path.join(path, "test.csv"))
train = generate_target(train)
convert_label = {'Computer Science':0,'Physics':1,'Mathematics':2,'Statistics':3,'Quantitative Biology':4,'Quantitative Finance':5}
train['convert_label'] = train.LABEL.map(convert_label)
df_train, df_val = train_test_split(train, test_size=0.33, random_state=42)
train.head(3)


# In[6]:


df_val.convert_label.value_counts()


# In[7]:


class Transformer_Dataset(Dataset):

    def __init__(self, text, target, tokenizer, max_len):
        self.text = text
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, item):
        text = str(self.text[item])
        target = self.target[item]

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
        'target': torch.tensor(target, dtype=torch.long)
        }

def create_data_loader(df, text, target, tokenizer, max_len, batch_size):
  ds = Transformer_Dataset(
    text=df[text].to_numpy(),
    target=df[target].to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=2,
    drop_last = True
  )

# Get model configuration.
print('Loading configuraiton...')
model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name, 
                                          num_labels=n_labels)

# Get model's tokenizer.
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

# Get the actual model.
print('Loading model...')
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, 
                                                           config=model_config)

# Load model to defined device.
model.to(device)
print('Model loaded to `%s`'%device)


# In[8]:


train_data_loader = create_data_loader(df_train,'TITLE', 'convert_label', tokenizer, max_len, batches)
val_data_loader = create_data_loader(df_val, 'TITLE', 'convert_label', tokenizer, max_len, batches)


# In[9]:


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
                                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

total_steps = len(train_data_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)


# In[10]:


from sklearn import metrics
def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples, batch_size, maxlen):
    model = model.train()
    losses = []
    acc = 0
    counter = 0
  
    for d in data_loader:
        input_ids = d["input_ids"].reshape(batch_size,maxlen).to(device)
        attention_mask = d["attention_mask"].to(device)
        target = d['target'].to(device)
        
        outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels = target)
        loss = outputs[0]
        logits = outputs[1]

        # preds = preds.cpu().detach().numpy()
        _, prediction = torch.max(outputs[1], dim=1)
        target = target.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(target, prediction)

        acc += accuracy
        losses.append(loss.item())
        
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        counter = counter + 1

    return acc / counter, np.mean(losses)

def eval_model(model, data_loader, device, n_examples,batch_size,maxlen):
    model = model.eval()
    losses = []
    acc = 0
    counter = 0
  
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].reshape(batch_size,maxlen).to(device)
            attention_mask = d["attention_mask"].to(device)
            target = d['target'].to(device)
            
            outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels = target)
            loss = outputs[0]
            logits = outputs[1]

            _, prediction = torch.max(outputs[1], dim=1)
            target = target.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            accuracy = metrics.accuracy_score(target, prediction)

            acc += accuracy
            losses.append(loss.item())
            counter += 1

    return acc / counter, np.mean(losses)


# In[11]:



get_ipython().run_cell_magic('time', '', "history = defaultdict(list)\nbest_accuracy = 0\n\nfor epoch in tqdm(range(epochs)):\n    print(f'Epoch {epoch + 1}/{epochs}')\n    print('-' * 10)\n\n    train_acc, train_loss = train_epoch(\n        model,\n        train_data_loader,     \n        optimizer, \n        device, \n        scheduler, \n        len(df_train),\n        batches,\n        max_len\n    )\n\n    print(f'Train loss {train_loss} Train accuracy {train_acc}')\n\n    val_acc, val_loss = eval_model(\n        model,\n        val_data_loader, \n        device, \n        len(df_val),\n        batches,\n        max_len\n    )\n\n    print(f'Val loss {val_loss} Val accuracy {val_acc}')\n    print()\n\n    history['train_acc'].append(train_acc)\n    history['train_loss'].append(train_loss)\n    history['val_acc'].append(val_acc)\n    history['val_loss'].append(val_loss)\n\n    if val_acc > best_accuracy:\n        torch.save(model.state_dict(), os.path.join(path, output_model))\n        best_accuracy = val_acc")


# In[13]:


def predict_sentiment(text):
    text = text

    encoded_review = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    max_length=max_len,
    return_token_type_ids=False,
    pad_to_max_length=False,
    return_attention_mask=True,
    return_tensors='pt',
    truncation=True
    )

    input_ids = pad_sequences(encoded_review['input_ids'], maxlen=max_len, dtype=torch.Tensor ,truncating="post",padding="post")
    input_ids = input_ids.astype(dtype = 'int64')
    input_ids = torch.tensor(input_ids) 

    attention_mask = pad_sequences(encoded_review['attention_mask'], maxlen=max_len, dtype=torch.Tensor ,truncating="post",padding="post")
    attention_mask = attention_mask.astype(dtype = 'int64')
    attention_mask = torch.tensor(attention_mask) 

    input_ids = input_ids.reshape(1,max_len).to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    outputs = outputs[0][0].cpu().detach()

    probs = F.softmax(outputs, dim=-1).cpu().detach().numpy().tolist()
    _, prediction = torch.max(outputs, dim =-1)
    # return class_names[prediction]
    return prediction
    # print("Positive score:", probs[1])
    # print("Negative score:", probs[0])
    # print("Neutral score", probs[2])
    # print(f'text: {text}')
    # print(f'target  : {class_names[prediction]}')


# In[19]:


df_val.head(10)


# In[24]:


testid = 19372
print(df_val.TITLE[testid])
predict_sentiment(df_val.TITLE[testid])


# In[25]:


df_val['pred']=df_val.TITLE.apply(lambda x: int(predict_sentiment(x)))


# In[28]:


confusion_matrix(df_val.convert_label, df_val.pred)


# In[29]:


print(classification_report(df_val.convert_label, df_val.pred))

