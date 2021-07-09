#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

import pickle
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import string


text_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data\\BBC_News_Summary\\BBC_News_Summary\\News_Articles'
smr_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data\\BBC_News_Summary\\BBC_News_Summary\\Summaries'
# https://www.kaggle.com/datajameson/topic-modelling-nlp-amazon-reviews-bbc-news


# Reference: https://appliedmachinelearning.blog/2019/12/31/extractive-text-summarization-using-glove-vectors/

# In[2]:


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


# In[3]:


rawdata = textdf.merge(smrdf, how='left',on=['docid','type']).rename(columns={"news_x": "news", "news_y": "summary"})
rawdata


# In[9]:


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def rem_ascii(s):
    return "".join([c for c in s if ord(c) < 128 ])
 
# Cleaning the text sentences so that punctuation marks, stop words and digits are removed.
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    processed = re.sub(r"\d+","",punc_free)
    return processed


# In[4]:


embeddings_index = wdembd.loadGloveModel('C:/ProgramData/Anaconda3/append_file/glove/glove.6B.100d.txt')


# In[16]:


sentences = sent_tokenize(rawdata.news[1])
cleaned_text = [rem_ascii(wdembd.basic_text_clean(sentence)) for sentence in sentences]
print(cleaned_text)


# ## Sentence Embeddings from Glove

# In[21]:


sentence_vectors = []
dim=100
for i in cleaned_text:
    if len(i) != 0:
        v = sum([embeddings_index.get(w, np.zeros((dim,))) for w in i.split()])/(len(i.split())+0.001)
    else:
        v = np.zeros((dim,))
    sentence_vectors.append(v)


# ## Similarity Matrix

# In[23]:


sim_mat = np.zeros([len(cleaned_text), len(cleaned_text)])
for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i != j:
            sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,dim),sentence_vectors[j].reshape(1,dim))[0,0]
sim_mat = np.round(sim_mat,3)
print(sim_mat)
 
# Creating the network graph
nx_graph = nx.from_numpy_array(sim_mat)
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(nx_graph)
nx.draw(nx_graph, with_labels=True, font_weight='bold')
nx.draw_networkx_edge_labels(nx_graph,pos,font_color='red')
plt.show()


# ## Text Rank Algorithm

# In[24]:


scores = nx.pagerank(nx_graph)
ranked_sentences = sorted(((scores[i],i) for i,s in enumerate(sentences)), reverse=True)
arranged_sentences = sorted(ranked_sentences[0:int(len(sentences)*0.5)], key=lambda x:x[1])
print("\n".join([sentences[x[1]] for x in arranged_sentences]))


# In[22]:


def embedding_similarity_pagerank_extraction(doc, dim, embedding_index, reduction_ratio = 1/3):
    sentences = sent_tokenize(doc)
    sentence_vectors = []
    cleaned_text = [wdembd.basic_text_clean(sentence) for sentence in sentences]

    for i in cleaned_text:
        if len(i) != 0:
            v = sum([embedding_index.get(w, np.zeros((dim,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((dim,))
        sentence_vectors.append(v)

    sim_mat = np.zeros([len(cleaned_text), len(cleaned_text)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,dim),sentence_vectors[j].reshape(1,dim))[0,0]
    sim_mat = np.round(sim_mat,3)

    nx_graph = nx.from_numpy_array(sim_mat)

    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i],i) for i,s in enumerate(sentences)), reverse=True)
    arranged_sentences = sorted(ranked_sentences[0:int(len(sentences)*reduction_ratio)], key=lambda x:x[1])

    output = [sentences[x[1]] for x in arranged_sentences]
    return output


# In[27]:


print("\n".join(embedding_similarity_pagerank_extraction(rawdata.news[0], 100, embeddings_index, 0.1)))

