#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
import os
sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_nlp_pkg')
sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_model_explain_pkg')
import nlpbasic.textClean as textClean
import nlpbasic.docVectors as DocVector
import nlpbasic.dataExploration as DataExploration
import nlpbasic.lda as lda
import nlpbasic.tfidf as tfidf

import model_explain.plot as meplot
# import model_explain.shap as meshap
import tensorflow as tf
from numpy import array,asarray,zeros
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Embedding
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import re
root_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data'


# ## Load Data 

# In[4]:


raw_data = pd.read_csv(os.path.join(root_path, "sentiment140_twitter.csv"), names=['target', 'id', 'date', 'flag', 'user', 'text'], header=None)
data = raw_data[raw_data.text.isnull()==False].drop_duplicates().reset_index(drop=True)
data['hashtag'] = data['text'].apply(lambda x: textClean.get_hashtag(x))
data['atuser'] = data['text'].apply(lambda x: re.findall(r"@(\w+)", x))
data['clean_tweet'] = data['text'].apply(lambda x: textClean.remove_string_startwith(x, '@'))
data['clean_tweet'] = data['clean_tweet'].apply(lambda x: textClean.remove_string_startwith(x, '#'))
data['clean_tweet'] = data['clean_tweet'].apply(lambda x: textClean.remove_string_startwith(x, 'http'))
data['label'] = np.where(data['target']==0, 0, 1)


# In[3]:


data.head(3)


# In[4]:


i = 175
print(data.text[i])
print("-----------")
# print(data.clean_tweet[i])
print("-----------")
print(data.clean_tweet[i])


# ## Explore Data
# We run Explore Data in sample dataset.

# In[13]:


import random
sample_size = 100000

sample_data = pd.concat([data.loc[data['hashtag'].str.len() > 0], data.loc[data['hashtag'].str.len() == 0].sample((sample_size - len(data.loc[data['hashtag'].str.len() > 0])), random_state = 3)]).reset_index(drop = True)
pos_tweet = list(x.split() for x in sample_data[sample_data['label']==1]['clean_tweet'])
neg_tweet = list(x.split() for x in sample_data[sample_data['label']==0]['clean_tweet'])
postop10tfidf = tfidf.get_top_n_tfidf_bow(pos_tweet, top_n_tokens = 30)
negtop10tfidf = tfidf.get_top_n_tfidf_bow(neg_tweet, top_n_tokens = 30)
print('top 30 negative review tfidf', negtop10tfidf)
print('top 30 positive review tfidf', postop10tfidf)


# In[14]:


top10_posfreq_list = DataExploration.get_topn_freq_bow(pos_tweet, topn = 10)
top10_negfreq_list = DataExploration.get_topn_freq_bow(neg_tweet, topn = 10)
print(top10_posfreq_list)
print(top10_negfreq_list)


# In[15]:


DataExploration.generate_word_cloud(pos_tweet)


# In[16]:


DataExploration.generate_word_cloud(neg_tweet)


# In[18]:


hashtag_list = list(sample_data.hashtag)
DataExploration.generate_word_cloud(hashtag_list)


# We didn't remove stop words, so the LDA does not work well, to do topic modelling, we need to remove stop words. But for sentiment analysis, it is better to keep all words.
# However, even for sentiment analysis, we need to set minimal words length, there we need to set it as 2.

# In[164]:


no_topics = 10
lda_allbow, bow_corpus, dictionary = lda.fit_lda(pos_tweet, num_topics = no_topics)
lda.lda_topics(lda_allbow)


# ## Prepare training/testing/validation dataset

# In[2]:


X = [x for x in data.clean_tweet]
y = np.array(data.label)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.125, random_state = 11)


# ## Word Embedding & RNN

# ### Glove Word Embedding

# In[6]:


# load the whole embedding into memory
embeddings_index = dict()
embedding_dim = 100 
# download glove word embedding first and then load it with the following code
f = open('C:/ProgramData/Anaconda3/append_file/glove/glove.6B.100d.txt', encoding = 'utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs
f.close
print('loaded %s word vectors.' % len(embeddings_index))


# In[7]:


max_length = int(np.percentile(data.clean_tweet.apply(lambda x: len(x.split())), 95))
# we also tried max length, but it cause overfitting

t = Tokenizer()
t.fit_on_texts(X_train)
# print("words with freq:", t.word_docs)

vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(X_train)
print('Encoding:\n', encoded_docs[0])
print('\nText:\n', list(X_train)[0])
print('\nWord Indices:\n', [(t.index_word[i], i) for i in encoded_docs[0]])
print('vocab size:', vocab_size)
train_padded_docs = pad_sequences(encoded_docs, maxlen = max_length, padding = 'post')


# In[8]:


# Initialize a matrix with zeros having dimensions equivalent to vocab size and 100
embedding_matrix = zeros((vocab_size, embedding_dim))
for word, idx_word in t.word_index.items():
    word_vector = embeddings_index.get(word)
    if word_vector is not None:
        embedding_matrix[idx_word] = word_vector
print('word:', t.index_word[1])
print('Embedding:\n', embedding_matrix[1])
print('length of embedding matrix is:', len(embedding_matrix))
print('vocabulary size is %s.' % vocab_size)


# In[93]:


from scipy.spatial import distance
def find_closest_embeddings(embedding):
    return sorted(embeddings_index.keys(), key = lambda word: distance.euclidean(embeddings_index[word], embedding))
find_closest_embeddings((embedding_matrix[t.word_index['welcome']]))


# In[9]:


encoded_val_doc = t.texts_to_sequences(X_val)
padded_val_doc = pad_sequences(encoded_val_doc, maxlen = max_length, padding = 'post')
encoded_test_doc = t.texts_to_sequences(X_test)
padded_test_doc = pad_sequences(encoded_test_doc, maxlen = max_length, padding = 'post')


# ### RNN

# In[16]:


model = Sequential(
    [
        Embedding(vocab_size, embedding_dim, weights = [embedding_matrix], input_length = max_length, trainable = False),
        Flatten(),
        Dense(embedding_dim, activation="relu", name="layer1"),
        Dense(1, activation = 'sigmoid', name="layer2")
        
    ]
)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
print(model.summary())


# In[65]:


history = model.fit(train_padded_docs, y_train, epochs = 10, verbose = 1, batch_size = 32, validation_data = (padded_val_doc, y_val))


# In[66]:


acc = history.history['acc']
print ("Accuracy history: ",acc)
val_acc = history.history['val_acc']
print("\nValidation history: ",val_acc)
loss = history.history['loss']
val_loss = history.history['val_loss']


# Tried different pre-processing, with more pre-processing, we loss more information, but with raw data we have http web link and @ or #, which are noise, so in the end, we remove the hashtag, userid start with @ and the http link.

# In[67]:


# plot loss rate and accuracy
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, color = 'r', label = 'Training acc')
plt.plot(epochs, val_acc, color = 'b', label = 'Validation acc')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.figure()
plt.plot(epochs, loss, color = 'r', label = 'Training loss')
plt.plot(epochs, val_loss, color = 'b', label = 'Validation loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()


# In[68]:


pred_test = model.predict(padded_test_doc)
rounded_pred_test = np.round(pred_test)
cf_matrix = confusion_matrix(y_test, rounded_pred_test)
meplot.cf_matrix_heatmap(cf_matrix)


# In[69]:


print(classification_report(y_test,rounded_pred_test))


# To avoid overfitting, use epochs = 4.

# In[17]:


history = model.fit(train_padded_docs, y_train, epochs = 2, verbose = 1, batch_size = 32, validation_data = (padded_val_doc, y_val))


# In[19]:


pred_test = model.predict(padded_test_doc)
rounded_pred_test = np.round(pred_test)
cf_matrix = confusion_matrix(y_test, rounded_pred_test)
meplot.cf_matrix_heatmap(cf_matrix)


# In[20]:


print(classification_report(y_test,rounded_pred_test))


# ### Model Explainability
# Reference: https://github.com/slundberg/shap

# #### Lime

# In[21]:


from lime import lime_text
from lime.lime_text import LimeTextExplainer

def predict_for_lime(text_array):
    encoded =t.texts_to_sequences(text_array)
    text_data = pad_sequences(encoded, maxlen=max_length,padding='post')
    pred=model.predict(text_data)
    return pred

# test the predicition function
print ("Verify if predictions are correct for the function")
print(predict_for_lime([X_test[0],X_test[533]]))
print(y_test[0], y_test[533])
#initilaize Lime for text
explainer = LimeTextExplainer(class_names=["Positive"])


# In[22]:


#Check explanation for a negative review
exp = explainer.explain_instance(str(X_test[0]), predict_for_lime, num_features=10, top_labels=1)
exp.show_in_notebook()


# In[23]:


#Check explanation for a negative review
exp = explainer.explain_instance(str(X_test[533]), predict_for_lime, num_features=10, top_labels=1)
exp.show_in_notebook()


# ### LSTM 

# In[10]:


from tensorflow.keras.layers import LSTM
lstm_m = Sequential(
    [
        Embedding(vocab_size, embedding_dim, weights = [embedding_matrix], input_length = max_length, trainable = False, name = 'embedding'),
        LSTM(embedding_dim, dropout=0.2, recurrent_dropout=0.2, name = 'lstm'),
        Dense(1, activation = 'sigmoid', name="layer2")
    ]
)
lstm_m.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
print(lstm_m.summary())


# In[16]:


history = lstm_m.fit(train_padded_docs, y_train, epochs = 10, verbose = 1, batch_size = 32, 
                     validation_data = (padded_val_doc, y_val))


# In[18]:


acc = history.history['acc']
print ("Accuracy history: ",acc)
val_acc = history.history['val_acc']
print("\nValidation history: ",val_acc)
loss = history.history['loss']
val_loss = history.history['val_loss']
# plot loss rate and accuracy
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, color = 'r', label = 'Training acc')
plt.plot(epochs, val_acc, color = 'b', label = 'Validation acc')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.figure()
plt.plot(epochs, loss, color = 'r', label = 'Training loss')
plt.plot(epochs, val_loss, color = 'b', label = 'Validation loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()


# In[19]:


pred_test = lstm_m.predict(padded_test_doc)
rounded_pred_test = np.round(pred_test)
cf_matrix = confusion_matrix(y_test, rounded_pred_test)
meplot.cf_matrix_heatmap(cf_matrix)


# In[28]:


from lime import lime_text
from lime.lime_text import LimeTextExplainer

def predict_for_lime(text_array):
    encoded =t.texts_to_sequences(text_array)
    text_data = pad_sequences(encoded, maxlen=max_length,padding='post')
    pred=lstm_m.predict(text_data)
    return pred

# test the predicition function
print ("Verify if predictions are correct for the function")
print(predict_for_lime([X_test[2],X_test[511]]))
print(y_test[2], y_test[511])
#initilaize Lime for text
explainer = LimeTextExplainer(class_names=["Positive"])


# In[26]:


#Check explanation for a negative review
exp = explainer.explain_instance(str(X_test[2]), predict_for_lime, num_features=10, top_labels=1)
exp.show_in_notebook()


# In[29]:


#Check explanation for a negative review
exp = explainer.explain_instance(str(X_test[511]), predict_for_lime, num_features=10, top_labels=1)
exp.show_in_notebook()


# ## DL & Transfer Learning with fastai
# some reference:
# 
# https://www.kaggle.com/maroberti/fastai-with-transformers-bert-roberta
# 
# https://www.kaggle.com/twhelan/covid-19-vaccine-sentiment-analysis-with-fastai
# 

# - multivariate classification
# - classification with hashtag
# - topic modelling with hashtag
# - topic modelling 
# - word embedding with original text
# - classification with pre-trained word embedding (DL)
