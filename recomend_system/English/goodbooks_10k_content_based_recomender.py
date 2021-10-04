#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import io
import os
import re
import sys
import tensorflow as tf
from pandasql import sqldf
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
root_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data\\goodbooks'
sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_nlp_pkg')

import nlpbasic.textClean as TextProcessing
import nlpbasic.docVectors as DocVector
import nlpbasic.dataExploration as dataExploration
import nlpbasic.lda as lda
import nlpbasic.tfidf as tfidf

pd.set_option('display.max_columns', 500)
print(tf.__version__)
pysqldf = lambda q: sqldf(q, globals())


# **Reference**
# 
# [Learn about collaborative filtering and weighted alternatng least square with tensorflow](https://fairyonice.github.io/Learn-about-collaborative-filtering-and-weighted-alternating-least-square-with-tensorflow.html)

# In[2]:


book_tags = pd.read_csv(os.path.join(root_path, "book_tags.csv"))
books = pd.read_csv(os.path.join(root_path, "books.csv"))
ratings = pd.read_csv(os.path.join(root_path, "ratings.csv"))
tags = pd.read_csv(os.path.join(root_path, "tags.csv"))
to_read = pd.read_csv(os.path.join(root_path, "to_read.csv"))

#load ratings latent variable dataset
user_latent = pd.read_csv(os.path.join(root_path, 'user_latent.csv'))
item_latent = pd.read_csv(os.path.join(root_path, 'item_latent.csv'))


# In[3]:


def change_id_from_0(data, column):
    data_copy = data.copy()
    for i in column:
        data_copy[i] = data_copy[i] - 1
    return data_copy

book_tags = change_id_from_0(book_tags, ['goodreads_book_id'])
books = change_id_from_0(books, ['id'])
ratings = change_id_from_0(ratings, ['book_id','user_id'])
to_read = change_id_from_0(to_read, ['book_id','user_id'])
        
user_latent.columns = ['id','u_latent_1','u_latent_2','u_latent_3','u_latent_4',
                       'u_latent_5','u_latent_6','u_latent_7','u_latent_8',
                       'u_latent_9','u_latent_10','u_latent_11','u_latent_12',
                       'u_latent_13','u_latent_14','u_latent_15']
item_latent.columns = ['id','i_latent_1','i_latent_2','i_latent_3','i_latent_4',
                       'i_latent_5','i_latent_6','i_latent_7','i_latent_8',
                       'i_latent_9','i_latent_10','i_latent_11','i_latent_12',
                       'i_latent_13','i_latent_14','i_latent_15']


# In[4]:


print('--------------book_tags--------------')
display(book_tags.head(3))
print('--------------books--------------')
display(books.head(3))
print('--------------ratings--------------')
display(ratings.head(3))
print('--------------tags--------------')
display(tags.head(3))
print('--------------to_read--------------')
display(to_read.head(3))
print('--------------user_latent--------------')
display(user_latent.head(3))
print('--------------item_latent--------------')
display(item_latent.head(3))


# In[5]:


def count_N_unique(data, column):
    n_unique = len(data[column].unique())
    print(column, ":", n_unique)

print('All variables unique value count in books:')
for i in books.columns:
    count_N_unique(books, i)


# In[6]:


i = 7
print(ratings[ratings.user_id == i])
print(to_read[to_read.user_id == i])


# In[7]:


query = """
select A.*
    , B.original_title
    , B.authors

from ratings A
left join books B on A.book_id = B.book_id
"""
test = pysqldf(query)
test.head(3)


# In[8]:


stats = ratings.describe()
stats


# ## Content-Based Recomender System

# In[9]:


content_data = books[['id','authors','original_title','language_code','average_rating']]
content_data['doc'] = content_data['authors'] + ' ' + content_data['original_title'] + ' ' + content_data['language_code'] + ' ' + content_data['average_rating'].astype(str)
content_data = content_data.dropna()
content_data.head(3)


# In[10]:


processed_doc = TextProcessing.pipeline(content_data['doc'].to_list(), 
                                        multi_gram = [1,2], 
                                        lower_case=True, 
                                        deacc=False, encoding='utf8',
                                        errors='strict', 
                                        stem_lemma = 'lemma', 
                                        tag_drop = ['J'], 
                                        nltk_stop=True, 
                                        stop_word_list=['course','courses'], 
                                        check_numbers=False, 
                                        word_length=0, 
                                        remove_consecutives=True)


# In[163]:


dataExploration.generate_word_cloud(processed_doc)


# In[11]:


tfidf_value_data = tfidf.get_tfidf_dataframe(processed_doc,no_below =2, no_above = 1)
tfidf_value_data.head(10)


# In[12]:


base_book = 'To Kill a Mockingbird'
base_book_detail = content_data[content_data.original_title == base_book]
bookid = base_book_detail['id'].values
filter_data = tfidf_value_data[tfidf_value_data.doc_id.isin(bookid)]

test = dataExploration.get_similarity_cosin(tfidf_value_data, 
                                            filter_data, 
                                            'bow', doc_key = 'doc_id', filterbase='base')#, comp_col = 'tfidf_value', topn_output = 10)
recommendation = content_data[content_data.index.isin(test.baseindex.to_list())]
print('Base Book Details')
display(base_book_detail)
print('Recomended Book Details')
display(recommendation)

