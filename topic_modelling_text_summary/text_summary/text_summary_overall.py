#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import os
sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_nlp_pkg')
sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_model_explain_pkg')
import nlpbasic.textClean as textClean
import nlpbasic.docVectors as DocVector
import nlpbasic.dataExploration as DataExploration
import nlpbasic.lda as lda
import nlpbasic.tfidf as tfidf
import nlpbasic.text_summarize as txtsmr
import nlpbasic.word_embedding as wdembd

import model_explain.plot as meplot
import model_explain.shap as meshap

import data_visualization.distribution_plot as dbplot

from numpy import array,asarray,zeros
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Flatten,Embedding
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import re
import pycountry
import seaborn as sns
import plotly.express as px

import matplotlib
import matplotlib.dates as mdates


text_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data\\BBC_News_Summary\\BBC_News_Summary\\News_Articles'
smr_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data\\BBC_News_Summary\\BBC_News_Summary\\Summaries'
# https://www.kaggle.com/datajameson/topic-modelling-nlp-amazon-reviews-bbc-news


# In[3]:


folders=["business","entertainment","politics","sport","tech"]
x=[]
y=[]
z=[]

for i in folders:
    files=os.listdir(os.path.join(text_path, i))
    for text_file in files:
        file_path=os.path.join(os.path.join(text_path, i), text_file)
        with open(file_path,'rb') as f:
            data=f.read().decode('iso-8859-1')
        x.append(data)
        y.append(i)
        z.append(i+text_file[:3])
        
data={'news':x,'type':y, 'docid':z}
textdf = pd.DataFrame(data)

folders=["business","entertainment","politics","sport","tech"]
x=[]
y=[]
z=[]

for i in folders:
    files=os.listdir(os.path.join(smr_path, i))
    for text_file in files:
        file_path=os.path.join(os.path.join(smr_path, i), text_file)
        with open(file_path,'rb') as f:
            data=f.read()
        x.append(data)
        y.append(i)
        z.append(i+text_file[:3])
        
data={'news':x,'type':y, 'docid':z}
smrdf = pd.DataFrame(data)


# In[4]:


rawdata = textdf.merge(smrdf, how='left',on=['docid','type']).rename(columns={"news_x": "news", "news_y": "summary"})
rawdata


# In[5]:


def print_article(data, index_id):
    printdata = data[data.index==index_id]
    print('Type:',printdata.type.item(), '// docid:', printdata.docid.item())
    print('-------------------- Summary --------------------')
    print('Description:',printdata.summary.item())
    print('-------------------- News --------------------')
    print(printdata.news.item())

print_article(rawdata,0)


# ## LSA

# In[6]:


lsa_smr = txtsmr.lsa_text_extraction(rawdata.news[0], smooth = 0.4, topn = 6)
print("\n".join(lsa_smr))


# ## Glove similarity

# In[7]:


embeddings_index = wdembd.loadGloveModel('C:/ProgramData/Anaconda3/append_file/glove/glove.6B.100d.txt')
glv_smr = txtsmr.embedding_similarity_pagerank_extraction(rawdata.news[0], 100, embeddings_index)
print("\n".join(glv_smr))


# ## Package sumy

# In[8]:


import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


# In[9]:


parser = PlaintextParser.from_string(rawdata.news[1], Tokenizer("english"))
stemmer = Stemmer("english")
summarizer = Summarizer(stemmer)
summarizer.stop_words = get_stop_words("english")

for sentence in summarizer(parser.document, 6):
    print(sentence)


# In[10]:


testchinese = '温客行一眼就看出周子舒使用的是四季山庄的流云九宫步，狠狠教训了顾湘一顿，就带她离开了。张成岭看出周子舒有一身好武功，只是深藏不露，就主动过来和周子舒寒暄，还给他一块名帖，让他有事去镜湖山庄，张成岭着急给母亲买点心，就先行离开了。周子舒听到孩子们在唱那首五湖盟争夺武林盟主以及琉璃甲的歌谣，不禁感慨江湖的风云多变。周子舒叫醒岸边的摆渡船夫，他要乘船去镜湖山庄，摆渡船夫趁机狮子大开口，周子舒也不还价，摆渡船夫看他一副病恹恹的模样，不忍心敲诈他，温客行带顾湘及时赶来，主动提出送周子舒去镜湖山庄，摆渡船夫不依不饶，拉起周子舒就上船离开了。周子舒远远就发现镜湖山庄犹如人间仙境，他迫不及待赶过去，下船就忘了付钱，遭到摆渡船夫劈头盖脸一顿臭骂，周子舒索性就坐一次霸王船。周子舒施展轻功，很快就进入镜湖山庄的桃林，他沉醉于花香之中，温客行突然从背后偷袭，周子舒只能迎战，两个人交手几个回合，温客行对周子舒心生佩服，请他喝酒小叙，周子舒断然拒绝。周子舒来到镜湖山庄，从管家口中得知镜湖派掌门张玉森久不闻江湖事，他有三个儿子张成峰，张成峦和张成岭，也不许他们掺和江湖门派之争，管家把周子舒安顿到柴房，子时的时候，三秋钉又准时开始发作，周子舒感觉浑身疼痛难忍，只能发动全部功力为自己疗伤，突然听到外面人声嘈杂。周子舒打开门发现镜湖山庄已经变成一片火海，他飞身上屋顶观察，发现带着鬼面具的人在镜湖山庄大肆烧杀抢掠，怀疑是鬼谷的人所为，他立刻下去救人，张玉森，张成峦和张成峰父子三人被抓走，镜湖山庄的人几乎全部被杀，尸横遍野。摆渡船夫保护着张成岭想逃走，被鬼谷的人追杀，周子舒出手相救，掩护着他们俩乘船离开，远远看到温客行坐在华亭伤看热闹。周子舒把摆渡船夫和张成岭带到一间破庙，摆渡船夫说明张玉森曾经救过他的命，他在镜湖山庄门前摆渡三年，就是想等有朝一日报恩，摆渡船夫让张成岭去太湖找三白大侠，张成岭坚决不走。外面阴风阵阵，一群带鬼面具的人冲进来，一个自称吊死鬼的人叫嚣着进来抓张成岭，周子舒因为体力耗尽要静养半个时辰，摆渡船夫和吊死鬼战在一处，他渐渐体力不支被打翻在地，吊死鬼要杀了周子舒，张成岭拼命保护他，顾湘及时赶来，她和黑白无常大打出手，吊死鬼想杀张成岭，摆渡船夫奋不顾身护住他，被打成重伤。顾湘被恶鬼们团团包围，周子舒挣扎着跳起来为顾湘解围，把恶鬼们全部打跑，他因体力不支差点晕倒，温客行赶来抱住周子舒。摆渡船夫因为失血过多奄奄一息，温客行用内力帮他维持，船夫拜托周子舒把张成岭交给五湖盟的赵敬，还让张成岭当场给周子舒跪下磕头，周子舒满口答应，摆渡船夫说完这些话就咽气了。周子舒帮张成岭把摆渡船夫埋葬，张成岭累得精疲力尽，周子舒打算休息一夜再上路，温客行让顾湘生火，把干粮烤了给周子舒和张成岭，周子舒借口不饿不想吃，顾湘对他冷嘲热讽，张成岭也不吃顾湘的干粮，遭到顾湘的训斥，谴责他不知道报恩，张成岭连连向她赔礼道歉。温客行发现张成岭身受重伤，主动提出帮他医治，周子舒坚决不同意，两个人一言不合就大打出手。'
parser = PlaintextParser.from_string(testchinese, Tokenizer("chinese"))
stemmer = Stemmer("chinese")
summarizer = Summarizer(stemmer)
summarizer.stop_words = get_stop_words("chinese")

for sentence in summarizer(parser.document, 6):
    print(sentence)


# In[ ]:




