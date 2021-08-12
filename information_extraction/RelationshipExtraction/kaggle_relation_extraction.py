#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
import os
from spacy.matcher import Matcher 
from spacy.tokens import Span 

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm
datapath = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data'

pd.set_option('display.max_colwidth', 200)
get_ipython().run_line_magic('matplotlib', 'inline')


# This code used to generate sample files from Kaggle wikipedia dataset, the dataset path is: https://www.kaggle.com/mikeortman/wikipedia-sentences.
# 
# The reference of this code is: https://www.kaggle.com/pavansanagapati/knowledge-graph-nlp-tutorial-bert-spacy-nltk/notebook
# 
# ```python
# with open("/kaggle/input/wikipedia-sentences/wikisent2.txt") as myfile:
#     head = [next(myfile) for x in range(5000)]
# print(head[-1])
# textfile = open("wiki_5000.txt", "w")
# for element in head:
#     textfile.write(element + "\n")
# textfile.close()
# ```

# In[2]:


doc = nlp("the drawdown process is governed by astm standard d823")

for tok in doc:
    print(tok.text, "...", tok.dep_)


# In[3]:


def get_entities(sent):
    ## chunk 1
    ent1 = ""
    ent2 = ""
    
    prv_tok_dep = ""    # dependency tag of previous token in the sentence
    prv_tok_text = ""   # previous token in the sentence

    prefix = ""
    modifier = ""

  #############################################################
  
    for tok in nlp(sent):
        ## chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
            # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " "+ tok.text
      
    # check: token is a modifier or not
        if tok.dep_.endswith("mod") == True:
            modifier = tok.text
            # if the previous word was also a 'compound' then add the current word to it
            if prv_tok_dep == "compound":
                modifier = prv_tok_text + " "+ tok.text
      
    ## chunk 3
        if tok.dep_.find("subj") == True:
            ent1 = modifier +" "+ prefix + " "+ tok.text
            prefix = ""
            modifier = ""
            prv_tok_dep = ""
            prv_tok_text = ""      

    ## chunk 4
        if tok.dep_.find("obj") == True:
            ent2 = modifier +" "+ prefix +" "+ tok.text
    
    ## chunk 5  
    # update variables
        prv_tok_dep = tok.dep_
        prv_tok_text = tok.text
  #############################################################

    return [ent1.strip(), ent2.strip()]


# In[4]:


get_entities("the drawdown process is governed by astm standard d823")


# In[23]:


with open(os.path.join(datapath, "wiki_5000.txt")) as myfile:
    wiki_sample = [next(myfile) for x in range(10000)]
wiki_sample = pd.DataFrame(wiki_sample).rename(columns={0:'sentence'})
wiki_sample = wiki_sample[wiki_sample['sentence']!='\n'].reset_index(drop=True)
wiki_sample.head(3)


# In[25]:


i=0
print(wiki_sample.sentence[i])
get_entities(wiki_sample.sentence[i])


# In[26]:


entity_pairs = []

for i in tqdm(wiki_sample["sentence"]):
    entity_pairs.append(get_entities(i))


# In[28]:


entity_pairs[0]


# In[39]:


def get_relation(sent):
    doc = nlp(sent)

    # Matcher class object 
    matcher = Matcher(nlp.vocab)

    #define the pattern 
    pattern1 = [{'DEP':'ROOT'}]
    pattern2 = [{'DEP':'prep'},{'OP':"?"}]
    pattern3 = [{'DEP':'agent'},{'OP':"?"}]  
    pattern4 = [{'POS':'ADJ'},{'OP':"?"}] 

    matcher.add("matching_1", [pattern1, pattern2, pattern3, pattern4]) 

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]] 

    return(span.text)


# In[53]:


i = 125
print(wiki_sample.sentence[i])
print(entity_pairs[i])
get_relation(wiki_sample.sentence[i])


# In[54]:


relations = [get_relation(i) for i in tqdm(wiki_sample['sentence'])]


# In[55]:


# extract subject
source = [i[0] for i in entity_pairs]

# extract object
target = [i[1] for i in entity_pairs]

kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})


# In[56]:


# create a directed-graph from a dataframe
G=nx.from_pandas_edgelist(kg_df, "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())


# In[67]:


kg_df.head(3)


# In[57]:


plt.figure(figsize=(12,12))

pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
plt.show()


# In[66]:


G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="in World"], "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
plt.show()

