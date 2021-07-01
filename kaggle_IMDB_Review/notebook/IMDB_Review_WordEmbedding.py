#!/usr/bin/env python
# coding: utf-8

# In[71]:


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
import model_explain.shap as meshap

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
root_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data'


# In[6]:


raw_data = pd.read_csv(os.path.join(root_path, "IMDB Dataset.csv"))
data = raw_data[raw_data.review.isnull()==False]
data['label'] = np.where(data['sentiment']=='positive', 1,0)
data = data.drop_duplicates()
data.insert(0, 'index', data.index + 1)


# In[53]:


#Clean data using Bumblebee Pipeline
preprocessed_text = textClean.pipeline(data['review'][0:1000].to_list(), multi_gram = [1], lower_case=True, deacc=False, encoding='utf8',
                                           errors='strict', stem_lemma = 'lemma', tag_drop = ['V'], nltk_stop=True, stop_word_list=['movie','film','movies','films'], 
                                           check_numbers=True, word_length=3, remove_consecutives=True)
preprocessed_text = [' '.join(i) for i in preprocessed_text]


# In[9]:


data.review[0]


# ## Cleaned Text

# In[10]:


preprocessed_text[0]


# In[54]:


clean_data = pd.DataFrame({'review': preprocessed_text, 'label': data['label'][0:1000]})
clean_data['length'] = clean_data['review'].apply(lambda x: len(x.split()))
clean_data.length.hist(bins = 50)


# From the above histogram, we need to find the max length that could cover majority of the reviews, this could between 100 and 200.

# In[14]:


max_length = 180


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(clean_data['review'], clean_data['label'], test_size = 0.33, random_state = 42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)


# ## Neural Network
# ### Keras 101

# https://keras.io/about/
# 
# https://www.youtube.com/playlist?list=PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU
# 
# Keras is a deep learning framework written in Python, running on top of the machine learning platform TensorFlow.
# 
# **Important terms to understand first:**
# 
# - Layers: The basic building blocks for a NN.
#     - Dense
#     - Flatten
#     - Embedding
# - Model: Groups Layers into one full architecture
#     - Sequential
#         - Functional API
# - Padding: https://www.tensorflow.org/guide/keras/masking_and_padding
#     - We need to pad the sequences because they come with different lengths
#     - (Explore later: RNN can handle this situation)

# #### Prepare training reviews for modelling

# In[56]:


t = Tokenizer()
t.fit_on_texts(X_train)
# print("words with freq:", t.word_docs)

vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(X_train)
print('Encoding:\n', encoded_docs[0])
print('\nText:\n', list(X_train)[0])
print('\nWord Indices:\n', [(t.index_word[i], i) for i in encoded_docs[0]])

padded_docs = pad_sequences(encoded_docs, maxlen = max_length, padding = 'post')
# for sequences shorter than maxlen add 0 to the end of the sequence (when pdding = 'post'), for sequence longer than maxlen, trunc senquence from the end of the sequence


# #### Apply preprocessing on reviews of validation set and test set

# In[57]:


encoded_val_doc = t.texts_to_sequences(X_val)
padded_val_docs = pad_sequences(encoded_val_doc, maxlen = max_length, padding = 'post')
print("===== validation set =====")
print(padded_val_docs)

encoded_test_doc = t.texts_to_sequences(X_test)
padded_test_docs = pad_sequences(encoded_test_doc, maxlen = max_length, padding = 'post')
print("===== test set =====")
print(padded_test_docs)


# ### Word Embedding: GloVe

# Download glove word embedding from the site: https://nlp.stanford.edu/projects/glove/. There are multiple files with different vocabulary and tokens used in training. They can be really big, you can download any. Here I am using the smaller one. Download, unzip it and provide the path of file with 100d in it in the following code. 100d stands for 100 dimensions of the word embedding.

# In[58]:


# load the whole embedding into memory
embeddings_index = dict()
# download glove word embedding first and then load it with the following code
f = open('C:/ProgramData/Anaconda3/append_file/glove/glove.6B.100d.txt', encoding = 'utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs
f.close
print('loaded %s word vectors.' % len(embeddings_index))


# - The previous step extract all the words from a very huge corpus, we need to pull out only those words that are present in our vocabulary.
# - Let's load the required vectors for words into an embedding matrix

# In[59]:


# Initialize a matrix with zeros having dimensions equivalent to vocab size and 100
embedding_matrix = zeros((vocab_size, 100))
for word, idx_word in t.word_index.items():
    word_vector = embeddings_index.get(word)
    if word_vector is not None:
        embedding_matrix[idx_word] = word_vector
print('word:', t.index_word[1])
print('Embedding:\n', embedding_matrix[1])
print('length of embedding matrix is:', len(embedding_matrix))
print('vocabulary size is %s.' % vocab_size)


# #### Power of word embeddings:
# - word embeddings allow us to capture synonyms and similar words

# In[60]:


def find_closest_embeddings(embedding):
    return sorted(embeddings_index.keys(), key = lambda word: distance.euclidean(embeddings_index[word], embedding))


# In[40]:


distance.euclidean(embeddings_index['book'], embedding_matrix[t.word_index['book']])


# In[42]:


find_closest_embeddings(embedding_matrix[t.word_index['book']])


# ### Neural Netword Model

# - Let's initialized the model, we have one output node with sigmoid as a non linear function for classification
# - Embedding layer is not trainable as we are using pre-trained Glove embeddings
# - Embedding layer is actualy a kind of dictionary which is mapping integer indices of words to dense vectors

# In[61]:


model = Sequential(
    [
        Embedding(vocab_size, 100, weights = [embedding_matrix], input_length = max_length, trainable = False),
        Flatten(),
        Dense(100, activation="relu", name="layer1"),
        Dense(1, activation = 'sigmoid', name="layer2")
    ]
)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
print(model.summary())


# In[62]:


# fit the model
history = model.fit(padded_docs, y_train, epochs = 10, verbose = 1, batch_size = 32, validation_data = (padded_val_docs, y_val))


# In[63]:


#save model:
# model.save_weights('imdb_sequential_model1.h5')


# In[65]:


acc = history.history['acc']
print ("Accuracy history: ",acc)
val_acc = history.history['val_acc']
print("\nValidation history: ",val_acc)
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[68]:


# plot loss rate and accuracy
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, color = 'r', label = 'Training acc')
plt.plot(epochs, val_acc, color = 'b', label = 'Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, color = 'r', label = 'Training loss')
plt.plot(epochs, val_loss, color = 'b', label = 'Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()


# Evaluate model in testing dataset

# In[74]:


predsTest = model.predict(padded_test_docs)
roundedPredsTest = np.round(predsTest)
print('Confusion Matrix: Positive is class 1 and Negative is class 0')
cf_matrix = confusion_matrix(y_test, roundedPredsTest, labels = [1,0])
print(cf_matrix)
meplot.cf_matrix_heatmap(cf_matrix)


# In[75]:


print(classification_report(y_test,roundedPredsTest))


# ### Explainability

# #### Lime

# In[77]:


from lime import lime_text
from lime.lime_text import LimeTextExplainer

def predict_for_lime(text_array):
    encoded =t.texts_to_sequences(text_array)
    text_data = pad_sequences(encoded, maxlen=max_length,padding='post')
    pred=model.predict(text_data)
    return pred

# test the predicition function
print ("Verify if predictions are correct for the function")
print(predict_for_lime([X_train[13],X_train[533]]))

#initilaize Lime for text
explainer = LimeTextExplainer(class_names=["Positive"])


# In[79]:


#Check explanation for a negative review
exp = explainer.explain_instance(str(X_train[22]), predict_for_lime, num_features=10, top_labels=1)
exp.show_in_notebook()


# In[80]:


#Check explanation for a positive review
exp = explainer.explain_instance(str(X_train[269]), predict_for_lime, num_features=10, top_labels=1)
exp.show_in_notebook()


# #### SHAP

# ### References
# [1] Book: Deep learning with Python, https://www.manning.com/books/deep-learning-with-python
# 
# [2] LIME documentation for text module, https://lime-ml.readthedocs.io/en/latest/index.html
