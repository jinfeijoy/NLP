#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jinfeijoy/NLP/blob/main/kaggle_IMDB_Review/notebook/IMDB_fastai1_with_transformer_googlecolab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


#Please select GPU first (from Edit->NotebookSetting)
import pandas as pd
import numpy as np
import io
import os
import re
from google.colab import drive
get_ipython().system('pip install fastai==1.0.58')
# !pip install urllib3==1.25.4
get_ipython().system('pip install transformers==2.5.1')

drive.mount('/content/drive')
import transformers
import torch
import torch.optim as optim
import random 
from fastai import *
from fastai.text import *
from fastai.callbacks import *


# In[1]:





# # Load Data
# Reference: https://www.kaggle.com/maroberti/fastai-with-transformers-bert-roberta

# In[2]:


path = '/content/drive/MyDrive/colab_data'
def de_emojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')
def tweet_proc(df, text_col='text'):
    df['orig_text'] = df[text_col]
    # Remove twitter handles
    df[text_col] = df[text_col].apply(lambda x:re.sub('@[^\s]+','',x))
    # Remove URLs
    df[text_col] = df[text_col].apply(lambda x:x.replace('<br />', ' '))
    return df[df[text_col]!='']


# In[34]:


basic_tweet = pd.read_csv(os.path.join(path, "IMDB_Dataset.csv"))
basic_tweet = basic_tweet[basic_tweet.sentiment!='empty'].drop_duplicates().sample(1000, random_state = 10).reset_index(drop=True)
# basic_tweet = pd.read_csv(os.path.join(path, "tweet_dataset.csv"))
# basic_tweet = basic_tweet[basic_tweet.sentiment!='empty'].drop_duplicates().reset_index(drop=True)
# basic_tweet = basic_tweet[['sentiment','new_sentiment','old_text']].rename(columns={'old_text':'text', 'sentiment':'emotion', 'new_sentiment':'sentiment'})
basic_tweet = tweet_proc(basic_tweet,'review').dropna(subset=['sentiment'])
print(len(basic_tweet))
basic_tweet.head(3)


# # Main transformers classes
# * A **model class** to load/store a particular pre-train model.
# * A **tokenizer class** to pre-process the data and make it compatible with a particular model.
# * A **configuration class** to load/store the configuration of a particular model.
# 
# Pre-trained model name can be found here: https://huggingface.co/transformers/pretrained_models.html#pretrained-models

# In[35]:


from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig


# In[36]:


MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
    'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
    'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig)
}


# In[6]:


seed = 42
use_fp16 = False
bs = 16

# model_type = 'roberta'
# pretrained_model_name = 'roberta-base'

# model_type = 'bert'
# pretrained_model_name='bert-base-uncased'

model_type = 'distilbert'
pretrained_model_name = 'distilbert-base-uncased'

#model_type = 'xlm'
#pretrained_model_name = 'xlm-clm-enfr-1024'

# model_type = 'xlnet'
# pretrained_model_name = 'xlnet-base-cased'


# In[7]:


model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]


# In[8]:


# Function to set the seed for generating random numbers
def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


# In[9]:


seed_all(seed)


# # Data Pre-processing
# 
# ### Custom Tokenizer

# In[45]:


class TransformersBaseTokenizer(BaseTokenizer):
    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""
    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):
        self._pretrained_tokenizer = pretrained_tokenizer
        # self.max_seq_len = int(np.percentile(pretrained_tokenizer.max_len, 95))
        self.max_seq_len = pretrained_tokenizer.max_len
        self.model_type = model_type

    def __call__(self, *args, **kwargs): 
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length and add the spesial tokens"""
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        if self.model_type in ['roberta']:
            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]
            tokens = [CLS] + tokens + [SEP]
        else:
            tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
            if self.model_type in ['xlnet']:
                tokens = tokens + [SEP] +  [CLS]
            else:
                tokens = [CLS] + tokens + [SEP]
        return tokens


# In[46]:


transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)
fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules=[], post_rules=[])


# In this implementation, be carefull about 3 things :
# 
# 1. As we are not using RNN, we have to limit the sequence length to the model input size.
# 2. Most of the models require special tokens placed at the beginning and end of the sequences.
# 3. Some models like RoBERTa require a space to start the input string. For those models, the encoding methods should be called with `add_prefix_space` set to `True`.
# 
# Below, you can find the resume of each pre-process requirement for the 5 model types used in this tutorial. You can also find this information on the HuggingFace documentation in each model section.
# 
# `bert:       [CLS] + tokens + [SEP] + padding`
# 
# `roberta:    [CLS] + prefix_space + tokens + [SEP] + padding`
# 
# `distilbert: [CLS] + tokens + [SEP] + padding`
# 
# `xlm:        [CLS] + tokens + [SEP] + padding`
# 
# `xlnet:      padding + tokens + [SEP] + [CLS]`
# 
# ### Custom Numericalizer

# In[47]:


class TransformersVocab(Vocab):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(TransformersVocab, self).__init__(itos = [])
        self.tokenizer = tokenizer
    
    def numericalize(self, t:Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return self.tokenizer.convert_tokens_to_ids(t)
        #return self.tokenizer.encode(t)

    def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."
        nums = np.array(nums).tolist()
        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)
    
    def __getstate__(self):
        return {'itos':self.itos, 'tokenizer':self.tokenizer}

    def __setstate__(self, state:dict):
        self.itos = state['itos']
        self.tokenizer = state['tokenizer']
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})


# ### Custom processor

# In[48]:


transformer_vocab = TransformersVocab(tokenizer = transformer_tokenizer)
numericalize_processor = NumericalizeProcessor(vocab = transformer_vocab)
tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer, include_bos = False, include_eos = False)
transformer_processor = [tokenize_processor, numericalize_processor]


# ## Setting up the Databunch

# In[49]:


pad_first = bool(model_type in ['xlnet'])
pad_idx = transformer_tokenizer.pad_token_id


# In[50]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(basic_tweet, test_size=0.3, random_state=42)


# In[51]:


databunch = (TextList.from_df(train, cols='review', processor=transformer_processor)
             .split_by_rand_pct(0.1,seed=seed)
             .label_from_df(cols= 'sentiment')
             .add_test(test)
             .databunch(bs=bs, pad_first=pad_first, pad_idx=pad_idx))


# Check batch and tokenizer 

# In[35]:


print('[CLS] token :', transformer_tokenizer.cls_token)
print('[SEP] token :', transformer_tokenizer.sep_token)
print('[PAD] token :', transformer_tokenizer.pad_token)
databunch.show_batch()


# Check batch and numericalizer :

# In[36]:


print('[CLS] id :', transformer_tokenizer.cls_token_id)
print('[SEP] id :', transformer_tokenizer.sep_token_id)
print('[PAD] id :', pad_idx)
test_one_batch = databunch.one_batch()[0]
print('Batch shape : ',test_one_batch.shape)
print(test_one_batch)


# ### Custom model

# In[52]:


# defining our model architecture 
class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_model: PreTrainedModel):
        super(CustomTransformerModel,self).__init__()
        self.transformer = transformer_model
        
    def forward(self, input_ids, attention_mask=None):
        
        # attention_mask
        # Mask to avoid performing attention on padding token indices.
        # Mask values selected in ``[0, 1]``:
        # ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        attention_mask = (input_ids!=pad_idx).type(input_ids.type()) 
        
        logits = self.transformer(input_ids,
                                  attention_mask = attention_mask)[0]   
        return logits


# In[53]:


config = config_class.from_pretrained(pretrained_model_name)
config.num_labels = len(basic_tweet.sentiment.unique())
print(len(basic_tweet.sentiment.unique()))
config.use_bfloat16 = use_fp16
print(config)


# In[54]:


transformer_model = model_class.from_pretrained(pretrained_model_name, config = config)
custom_transformer_model = CustomTransformerModel(transformer_model = transformer_model)


# ## Learner : Custom Optimizer / Custom Metric

# In[55]:


from transformers import AdamW
from functools import partial

CustomAdamW = partial(AdamW, correct_bias=False)

learner = Learner(databunch, 
                  custom_transformer_model, 
                  opt_func = CustomAdamW, 
                  metrics=[accuracy, error_rate])#.to_fp16()

# Show graph of learner stats and metrics after each epoch.
learner.callbacks.append(ShowGraph(learner))

if use_fp16: learner = learner.to_fp16()


# In[23]:


print(learner.model)


# In[21]:


# list_layers = [learner.model.transformer.roberta.embeddings,
#               learner.model.transformer.roberta.encoder.layer[0],
#               learner.model.transformer.roberta.encoder.layer[1],
#               learner.model.transformer.roberta.encoder.layer[2],
#               learner.model.transformer.roberta.encoder.layer[3],
#               learner.model.transformer.roberta.encoder.layer[4],
#               learner.model.transformer.roberta.encoder.layer[5],
#               learner.model.transformer.roberta.encoder.layer[6],
#               learner.model.transformer.roberta.encoder.layer[7],
#               learner.model.transformer.roberta.encoder.layer[8],
#               learner.model.transformer.roberta.encoder.layer[9],
#               learner.model.transformer.roberta.encoder.layer[10],
#               learner.model.transformer.roberta.encoder.layer[11],
#               learner.model.transformer.roberta.pooler]
# learner.split(list_layers)
learner.freeze_to(-1)
# learner.summary()


# In[56]:


learner.unfreeze()
learner.lr_find()
learner.recorder.plot(skip_end=10,suggestion=True)


# In[57]:


learner.unfreeze()
# learner.freeze_to(-2)
learner.fit_one_cycle(3,max_lr=2e-03)


# In[58]:


learner.predict('this movie is boring')


# In[29]:


learner.export(file = os.path.join(path, 'transformer.pkl'))


# In[59]:


def get_preds_as_nparray(ds_type) -> np.ndarray:
    """
    the get_preds method does not yield the elements in order by default
    we borrow the code from the RNNLearner to resort the elements into their correct order
    """
    preds = learner.get_preds(ds_type)[0].detach().cpu().numpy()
    sampler = [i for i in databunch.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    return preds[reverse_sampler, :]

test_preds = get_preds_as_nparray(DatasetType.Test)


# In[60]:


test_preds


# In[ ]:


test['prediction'] = np.argmax(test_preds,axis=1)


# In[ ]:


test.head(3)


# In[ ]:


np.argmax(test_preds,axis=1)

