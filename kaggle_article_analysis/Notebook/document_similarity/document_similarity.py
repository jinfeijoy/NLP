#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import requests
import os
import pandas as pd
import numpy as np
sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_nlp_pkg')
sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_model_explain_pkg')
import nlpbasic.textClean as textClean
import nlpbasic.docVectors as DocVector
import nlpbasic.dataExploration as DataExploration
import nlpbasic.lda as lda
import nlpbasic.tfidf as tfidf
import model_explain.plot as meplot
import model_explain.shap as meshap
import nlpbasic.extract_text_from_url as extract_txt
import data_visualization.seaborn_plt as snsplt
import matplotlib.pyplot as plt
import seaborn as sns
data_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data\\BerkshireLetter'


# Extract text file from url link: (berkshirehathaway annual letter).
# 
# Reference: https://medium.com/analytics-vidhya/best-nlp-algorithms-to-get-document-similarity-a5559244b23b

# ## Data Process
# this step is to grab data from web, don't need to run after initial generation

# In[5]:


format1 = range(1977,1998,1)
format2 = [1998, 2000, 2001, 2002]
format3 = range(2003, 2021, 1)
for i in format1:
    filename = 'letter_' + str(i) +'.txt'
    filename = os.path.join(data_path, filename)
    print(filename)
    url_link = 'https://www.berkshirehathaway.com/letters/' + str(i) + '.html'
    extract_txt.extract_text_from_texturl(url_link, filename)
for i in format2:
    filename = 'letter_' + str(i) +'.txt'
    filename = os.path.join(data_path, filename)
    print(filename)
    url_link = 'https://www.berkshirehathaway.com/letters/' + str(i) + 'pdf.pdf'
    extract_txt.extract_text_from_pdfurl(url_link, filename)    
extract_txt.extract_text_from_pdfurl('https://www.berkshirehathaway.com/letters/final1999pdf.pdf', os.path.join(data_path, 'letter_1999.txt')) 
for i in format3:
    filename = 'letter_' + str(i) +'.txt'
    filename = os.path.join(data_path, filename)
    print(filename)
    url_link = 'https://www.berkshirehathaway.com/letters/' + str(i) + 'ltr.pdf'
    extract_txt.extract_text_from_pdfurl(url_link, filename)    


# In[2]:


def get_letter_dict(letters_path,init_year=1977, end_year=2020):
    letters_dict = dict()
    letters_years = [year for year in range(init_year, end_year + 1)]
    for year in letters_years:
        filename = 'letter_' + str(year) + '.txt'
        path = os.path.join(letters_path, filename)
        letter = open(path,'r+', encoding='utf8').read()
        letters_dict[year] = letter
    return letters_dict


# In[3]:


letters_dictionary = get_letter_dict(data_path,init_year=1977, end_year=2020)
letters = list(letters_dictionary.values())
yearid = list(letters_dictionary.keys())


# In[28]:


import re
re.sub("[^a-z0-9A-Z\,]", " ", letters[0]) #this is not perfect because it will keep b or n from n\ b\


# In[39]:


import pickle
with open(os.path.join(data_path,'letters_dict.pickle'), 'wb') as handle:
    pickle.dump(letters_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[21]:


frequent_words_list = ['year', 'business', 'berkshire', 'million', 'company', 'u',
                       'billion', 'tax', 'investment', 'share', 'last', 'value',
                       'would', 'many', 'operation', 'market', 'one', 'large', 'p',
                       'make', 'asset', 'also', 'see', 'even', 'well', 'two', 'made',
                       'pre', 'return', 'cost', 'capital', 'may', 'price', 'per', 'first',
                       'annual', 'though', 'however', 'time',
                       'manager', 'money', 'dollar', 'meeting',
                       'interest', 'great', 'group', 'come', 'far',
                       'long', 'worth', 'net', 'report', 'industry', 'put',
                       'contract', 'must', 'yearend', 'day', 'major', 'real', 'since',
                       'let', 'need', 'record', 'good', 'country', 'america', 'period',
                       'average', 'increased', 'home', 'run', 'way',
                       'world', 'second', 'four', 'product', 'largest', 'certain',
                       'financial', 'three',
                       'ago', 'almost', 'american',
                       'amount', 'area', 'b',
                       'believe', 'better', 'blue', 'board',
                       'book', 'borrower', 'brown', 'buffett', 'buyer',
                       'c', 'capacity', 'case', 'cash', 'casualty', 'cat', 'category',
                       'change', 'charge', 'chip', 'charlie',
                       'committee', 'common', 'controlled', 'corporate',
                       'corporation', 'could', 'coupon', 'customer', 'debt',
                       'director',
                       'every',
                       'fee', 'figure', 'find', 'five', 'float',
                       'fund', 'g',
                       'general', 'get', 'give', 'goodwill',
                       'h', 'hathaway', 'helper', 'high', 'holding',
                       'hour', 'housing', 'huge', 'important', 'inc', 'income',
                       'intrinsic',
                       'investor', 'k', 'know', 'le', 'like', 'line', 'look',
                       'low', 'management', 'medium', 'merger',
                       'mr', 'much', 'name', 'national', 'never',
                       'new', 'news', 'non', 'number', 'often', 'operating',
                       'others', 'owned', 'owner', 'ownership', 'page', 'paid',
                       'paper', 'past', 'pay', 'payment', 'people',
                       'preferred', 'premium',
                       'purchase', 'question',
                       'rather', 'ratio',
                       'reported', 'reserve', 'result', 'retained',
                       'rule', 'say',
                       'september', 'service', 'shoe',
                       'star', 'state', 'store', 'subsidiary', 'sunday', 'super',
                       'take', 'ten', 'th', 'eht',
                       'therefore', 'utility', 'volume',
                       'zero']


# In[22]:


preprocessed_text = textClean.pipeline(letters, multi_gram = [1], lower_case=True, deacc=False, encoding='utf8',
                                       errors='strict', stem_lemma = 'lemma', tag_drop = [], nltk_stop=True, 
                                       stop_word_list=frequent_words_list, remove_pattern = ['www'],
                                       check_numbers=True, word_length=2, remove_consecutives=True)
preprocessed_text = [' '.join(i) for i in preprocessed_text]


# In[23]:


processed_letter = {
    'year': yearid,
    'clean_letter': preprocessed_text
}
processed_letter_df = pd.DataFrame(processed_letter, columns = ['year', 'clean_letter'])
processed_letter_df.to_csv(os.path.join(data_path, 'processed_letter.csv'))


# ## Data Exploration

# In[4]:


processed_letter_df = pd.read_csv(os.path.join(data_path, 'processed_letter.csv'))
processed_letter_df['tokens'] = processed_letter_df.clean_letter.apply(lambda x: x.split(' '))


# In[26]:


top_10_freq_words = [i[0] for i in DataExploration.get_topn_freq_bow(processed_letter_df['tokens'].to_list(), topn = 10)]
print(top_10_freq_words)


# In[27]:


top30tfidf = tfidf.get_top_n_tfidf_bow(processed_letter_df['tokens'].to_list(), top_n_tokens = 30)
print('top 30 tfidf', top30tfidf)


# In[28]:


DataExploration.generate_word_cloud(processed_letter_df['tokens'].to_list())


# In[29]:


no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(processed_letter_df['tokens'].to_list(), num_topics = no_topics)
lda.lda_topics(lda_allbow)


# ### Words Frequency

# In[8]:


dictionary = DocVector.generate_corpus_dict(processed_letter_df['tokens'].to_list(), no_below =1,
                                            no_above = 1, keep_n = 100000)
bow_corpus = DocVector.create_document_vector(processed_letter_df['tokens'].to_list(), dictionary)
my_df = DocVector.get_vocab_matrix(bow_corpus, dictionary)


# In[31]:


test = my_df[top_10_freq_words]
test.index = yearid
test = test.T


# In[32]:


snsplt.plot_heatmap(test, x='year', y='count', title = 'Top 10 words heatmap')


# ## Doc Similarity
# ### tfidf

# In[5]:


tfidf_data = tfidf.get_tfidf_dataframe(processed_letter_df['tokens'].to_list(), doc_index = yearid,no_below =5, no_above = 0.5, keep_n = 100000)
tfidf_data.head(3)


# In[6]:


test = DataExploration.get_similarity_cosin(tfidf_data, tfidf_data[tfidf_data.doc_id ==2008], 'bow', 'doc_id')
test.head(15)


# ### word2vec

# In[9]:


max_length = 7922
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
t = Tokenizer()
t.fit_on_texts(processed_letter_df['tokens'])
# print("words with freq:", t.word_docs)

vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(processed_letter_df['tokens'])
# print('Encoding:\n', encoded_docs[0])
# print('\nText:\n', list(processed_letter_df['tokens'])[0])
# print('\nWord Indices:\n', [(t.index_word[i], i) for i in encoded_docs[0]])

padded_docs = pad_sequences(encoded_docs, maxlen = max_length, padding = 'post')
padded_docs = pd.DataFrame(padded_docs)
padded_docs.index = yearid
padded_docs.head(3)


# In[11]:


from scipy.spatial.distance import cosine
cosine(padded_docs.iloc[[0]], padded_docs.iloc[[1]])


# In[15]:


test = DataExploration.get_similarity_cosin(padded_docs, padded_docs[padded_docs.index==2008], 'bow', 'doc_id',dataformat = 'wide')
test.head(15)


# ## Transformers
# Please refer to document_similarity_gcolab, which run on google colab.

# In[16]:


from sentence_transformers import SentenceTransformer


# In[23]:


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# In[24]:


embedding = model.encode(processed_letter_df['tokens'])
embedding = pd.DataFrame(embedding)
embedding.index = yearid


# In[25]:


test = DataExploration.get_similarity_cosin(embedding, embedding[embedding.index==2008], 'bow', 'doc_id',dataformat = 'wide')
test.head(15)

