#!/usr/bin/env python
# coding: utf-8

# In[102]:


import pandas as pd
import numpy as np
import io
import os
import re
import sys
from pandasql import sqldf
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
import jieba
import json
root_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data\\chinese_recommender'
sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_nlp_pkg')
stop_words_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP\\recomend_system\\Chinese\\hit_stopwords.txt'

sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_nlp_pkg')
import nlpbasic.dataExploration as DataExploration
import nlpbasic.lda as lda
import nlpbasic.tfidf as tfidf
import nlpbasic.dataExploration as dataExploration


# In[3]:


weibo_data = dfs = pd.read_excel(os.path.join(root_path, "Weibo_2020Coron.xlsx"),sheet_name='results')
weibo_data['index'] = weibo_data.index
weibo_data.head(3)


# In[4]:


print('start read stopwords data.')
stopwords = []
with open(stop_words_path, 'r', encoding='utf-8') as f:
    for line in f:
        if len(line)>0:
            stopwords.append(line.strip())
print(stopwords)


# ## Process text and generate word dictionary

# In[38]:


def preprocess_text(text):
    try:
        segs=jieba.lcut(text)
        segs = filter(lambda x:len(x)>1, segs)
        segs = [v for v in segs if not str(v).isdigit()]#去数字
        segs = list(filter(lambda x:x.strip(), segs)) #去左右空格
        segs = filter(lambda x:x not in stopwords, segs)
        temp = " ".join(segs)
        return(temp)
    except Exception:
        pass


# In[110]:


weibo_data['tokenizer'] = weibo_data['title'].apply(lambda x: preprocess_text(x))
weibo_data.columns = ['date1', 'date2', 'title_translate', 'title', 'searchCount', 'coron', 'index', 'tokenizer']
weibo_data['date1'] = pd.to_datetime(weibo_data['date1'])
weibo_data['date2'] = pd.to_datetime(weibo_data['date2'])
weibo_data.head(3)


# In[47]:


document = list(weibo_data.tokenizer.dropna())
document[0:3]


# In[48]:


def getWordDict(document, data_path, min_count=5):
    """
    功能：构建单词词典
    """
    word2id = {}
    # 统计词频
    for word in sum([a_tuple.split(' ') for a_tuple in document],[]):
        if word2id.get(word) == None:
            word2id[word] = 1
        else:
            word2id[word] += 1
            
    # 过滤低频词
    vocab = set()
    for word,count in word2id.items():
        if count >= min_count:
            vocab.add(word)

    # 构成单词到索引的映射词典
    word2id = {"PAD":0,"UNK":1}
    length = 2
    for word in vocab:
        word2id[word] = length
        length += 1
    with open(os.path.join(data_path, "word2id.json"),'w',encoding="utf-8") as fp:
        json.dump(word2id,fp,ensure_ascii=False)
        
        
getWordDict(document, root_path, min_count=2)


# Or we can use gensim package to get dictionary: https://cloud.tencent.com/developer/article/1844858

# In[92]:


import pprint 
text = weibo_data['tokenizer']
sentences = [] 
for item in text:     
    sentence = str(item).split(' ')     
    sentences.append(sentence)
from gensim import corpora 
dictionary = corpora.Dictionary(sentences) 
print(dictionary)
print(dictionary.token2id)


# In[93]:


# Save dictionary
dictionary.save(os.path.join(data_path, 'mydic.dict'))
corpus = [dictionary.doc2bow(sentence) for sentence in sentences] 
corpora.MmCorpus.serialize('bow.mm', corpus)


# ## Word Cloud Plot

# In[84]:


# get word cloud plot
from numpy import array, concatenate, atleast_1d
pandemic_weibo = list(weibo_data[weibo_data.coron==1]['tokenizer'])
all_tokens = concatenate([atleast_1d(a) for a in pandemic_weibo])
all_tokens = " ".join(all_tokens)
img_link = os.path.join(root_path, 'virus.png')
font_path = "C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP\\recomend_system\\Chinese\\simsun.ttf"
DataExploration.generate_chinese_word_cloud(all_tokens, img_link, font_path ,stopwords = '', color_control = True, save = True)


# In[88]:


# get word cloud plot
from numpy import array, concatenate, atleast_1d
pandemic_weibo = list(c[weibo_data.coron!=1]['tokenizer'].dropna())
all_tokens = concatenate([atleast_1d(a) for a in pandemic_weibo])
all_tokens = " ".join(all_tokens)
img_link = os.path.join(root_path, 'weibo.png')
font_path = "C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP\\recomend_system\\Chinese\\simsun.ttf"
DataExploration.generate_chinese_word_cloud(all_tokens, img_link, font_path ,stopwords = '', color_control = True, save = True)


# In[132]:


# get word cloud plot
import datetime
from numpy import array, concatenate, atleast_1d
pandemic_weibo = list(weibo_data[weibo_data.date1>datetime.datetime(2020,4,1)]['tokenizer'].dropna())
all_tokens = concatenate([atleast_1d(a) for a in pandemic_weibo])
all_tokens = " ".join(all_tokens)
img_link = os.path.join(root_path, 'virus.png')
font_path = "C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP\\recomend_system\\Chinese\\simsun.ttf"
DataExploration.generate_chinese_word_cloud(all_tokens, img_link, font_path ,stopwords = '', color_control = True, save = True)


# In[127]:


weibo_data.date1.sample(10)


# ## TFIDF

# In[103]:


weibo_data.sample(10)


# In[101]:


corseid = list(weibo_data.index)
tfidf_data = tfidf.get_tfidf_dataframe(sentences, doc_index = corseid, no_below =1, no_above = 0.5, keep_n = 100000)


# In[106]:


test = dataExploration.get_similarity_cosin(tfidf_data[tfidf_data.doc_id ==7413], 
                                            tfidf_data, 
                                            'bow', 
                                            'doc_id', 
                                            index_is_int = False, 
                                            topn = 50)
test = test.merge(weibo_data[['index','title']], how = 'left', left_on = ['compareindex'], right_on = ['index']).drop(columns = ['index']).rename(columns={'display_name':'compare_name'})
test = test.merge(weibo_data[['index','title']], how = 'left', left_on = ['baseindex'], right_on = ['index']).drop(columns = ['index']).rename(columns={'display_name':'base_name'})
test.head(3)


# In[107]:


display(tfidf_data.head(3))
list(test.title_x)


# ## LDA

# In[133]:


lda_allbow, bow_corpus, dictionary = lda.fit_lda(sentences, num_topics = 5)


# In[134]:


lda.lda_topics(lda_allbow)


# In[135]:


import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
pyLDAvis.enable_notebook()


# In[137]:


lda_allbow, bow_corpus, dictionary = lda.fit_lda(sentences, num_topics = 5)
vis = gensimvis.prepare(lda_allbow, bow_corpus, dictionary)
vis

