#!/usr/bin/env python
# coding: utf-8

# # NLP Academy Day 4

# The IMDB Movie Dataset is a large benchmark dataset which contains 50,00 movie reviews that are classified as either positive or negative.
# 
# **Positive Example**
# 
# Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring. It just never gets old, despite my having seen it some 15 or more times in the last 25 years. Paul Lukas' performance brings tears to my eyes, and Bette Davis, in one of her very few truly sympathetic roles, is a delight. If I had a dozen thumbs, they'd all be "up" for this movie.
# 
# 
# **Negative Example**
# 
# Besides being boring, the scenes were oppressive and dark. The movie tried to portray some kind of moral, but fell flat with its message. What were the redeeming qualities?? On top of that, I don't think it could make librarians look any more unglamorous than it did.
# 
# Our task today will be to build an NLP pipeline to produce a *supervised* learning algorithm to properly classify the reviews into positive examples and negative examples.
# 
# The data can be found here: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# ## Library Imports

# We'll start with the same base of packages that was used in the labs on Day 2 & 3 - so you should have everything you need already installed. If you want to bring other packages to the party, however, you are more than welcome to do so.

# In[20]:


import sys
sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_nlp_pkg')
sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_model_explain_pkg')
import os
import pandas as pd
import re
import string
import warnings
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from gensim import corpora
from numpy import array,asarray,zeros

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
os.getcwd()
import nlpbasic.textClean as textClean
import nlpbasic.docVectors as DocVector
import nlpbasic.dataExploration as DataExploration
import nlpbasic.lda as lda
import nlpbasic.tfidf as tfidf
import model_explain.plot as meplot
import model_explain.shap as meshap

from gensim import models

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Flatten,Embedding

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
root_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data'


# ## Task 0: Import the Data

# In[3]:


raw_data = pd.read_csv(os.path.join(root_path, "IMDB Dataset.csv"))
data = raw_data[raw_data.review.isnull()==False]
data['label'] = np.where(data['sentiment']=='positive', 1,0)
data = data.drop_duplicates()
data.insert(0, 'index', data.index + 1)


# ## Task 1: Prepare the Data For Modelling

# Using standard NLP pre-processing techniques discussed in NLP Academy 1 & 2, prepare the movie review data sets for modelling

# In[4]:


preprocessed_tokens = textClean.pipeline(data['review'][0:1000].to_list(), multi_gram = [1], lower_case=True, deacc=False, encoding='utf8',
                                        errors='strict', stem_lemma = 'lemma', tag_drop = ['V'], nltk_stop=True, stop_word_list=['movie','film','movies','films'], 
                                        check_numbers=True, word_length=3, remove_consecutives=True)


# ## Task 2: Create a Term Document Matrix using TF-IDF

# During the Day 2 lab, we created a term-document matrix by simply counting the occurence of words in each document. Let's try using TF-IDF to turn our documents in vectors here.

# In[5]:


tfidf_value_data = tfidf.get_tfidf_dataframe(preprocessed_tokens)
to10_tfidf_bow = tfidf.get_top_n_tfidf_bow(preprocessed_tokens, top_n_tokens = 10)
to10_tfidf_bow


# In[6]:


dictionary = DocVector.generate_corpus_dict(preprocessed_tokens, no_below =1,
                                            no_above = 0.5, keep_n = 100000)
bow_corpus = DocVector.create_document_vector(preprocessed_tokens, dictionary)
tfidf_trans = models.TfidfModel(bow_corpus)
my_df = DocVector.get_vocab_matrix(tfidf_trans[bow_corpus], dictionary)


# In[7]:


my_df.head(3)


# ## Task 3: Use a Traditional Machine Learning Model to Classify the Documents

# You're free to use any traditional machine learning model that you like, but make sure that you follow the best practices in your model building pipeline that were covered in Day 2! (as time allows)
# 
# * Proper Evaluation Metrics
# * Cross Validation
# * Hyperparameter Tuning
# * Feature Selection
# * Model Interpretability

# In[8]:


X_train, X_test, y_train, y_test = train_test_split(my_df, data['label'][0:1000], test_size = 0.33, random_state = 11)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 11)


# In[75]:


X_train.head(3)
X_train[data['label'] == 1]


# Training the model using XGBoost

# In[9]:


import xgboost as xgb


# In[10]:


dtrain = xgb.DMatrix(X_train, label = y_train, feature_names = X_train.columns)
dval = xgb.DMatrix(X_val,  label = y_val,  feature_names = X_val.columns)
evallist = [(dtrain,'train'), (dval,'validation')]
# specify parameters via map
param = {'objective':'binary:logistic',
         'max_depth':6,
         'learning_rate': 0.3,
         'eval_metric': 'auc',
         'subsample': 1,
         'colsample_bytree': 1,
         'colsample_bylevel': 1,
         'reg_alpha': 0,                  # 0.0  (L1 regularization)
         'reg_lambda': 1,                 # 1.0  (L2 regularization)
         'min_split_loss': 0,
         'min_child_weight': 1,
         'silent': 1
        }

num_round = 100
early_stop_round = 50
train_monitor = dict()


# In[11]:


xgbmodel = xgb.train(param, 
                     dtrain, 
                     num_boost_round = num_round,
                     evals = evallist, 
                     verbose_eval = 10,
                     early_stopping_rounds = early_stop_round,
                     evals_result = train_monitor
                    )


# In[12]:


#Check Model Performance on the test set
dtest = xgb.DMatrix(X_test,  feature_names = X_test.columns)
y_pred = xgbmodel.predict(dtest)


# In[13]:


variable_importance = np.array(list(xgbmodel.get_score(importance_type='gain').values()))
meplot.plot_var_imp(variable_importance, X_train.columns, 20)


# In[16]:


j = 1
plot, shap_val = meshap.get_SHAP_plot(model = xgbmodel, X = X_val, model_type = 'regression', define_index = False, index_value = j)
print("Review:",data.review[X_val.iloc[[j]].index[0]])
plot


# ## Task 4: Load Pre-Trained Glove Word Embeddings

# First, point the 'path' variable to the location of the Glove embeddings that were sent in advance of this lab, then read in the Glove vectors line by line.

# In[24]:


# load the whole embedding into memory
embeddings_index = dict()
embedding_dim = 300 
# download glove word embedding first and then load it with the following code
f = open('C:/ProgramData/Anaconda3/append_file/glove/glove.6B.300d.txt', encoding = 'utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs
f.close
print('loaded %s word vectors.' % len(embeddings_index))


# In[18]:


# create train/test/val with minimal preprocessing
X = raw_data.review
y = np.array([1 if raw_data.sentiment[i] == 'positive' else 0 for i in range(len(raw_data.sentiment))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.125, random_state = 11)
             
train = [review.replace('<br />', ' ') for review in X_train]
test = [review.replace('<br />', ' ') for review in X_test]
val = [review.replace('<br />',' ') for review in X_val]


# In[22]:


max_length = int(np.percentile(X.apply(lambda x: len(x.split())), 95))

t = Tokenizer()
t.fit_on_texts(train)
# print("words with freq:", t.word_docs)

vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(train)
print('Encoding:\n', encoded_docs[0])
print('\nText:\n', list(X_train)[0])
print('\nWord Indices:\n', [(t.index_word[i], i) for i in encoded_docs[0]])

padded_docs = pad_sequences(encoded_docs, maxlen = max_length, padding = 'post')
# for sequences shorter than maxlen add 0 to the end of the sequence (when pdding = 'post'), for sequence longer than maxlen, trunc senquence from the end of the sequence


# In[23]:


vocab_size


# In[25]:


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


# In[30]:


from scipy.spatial import distance
def find_closest_embeddings(embedding):
    return sorted(embeddings_index.keys(), key = lambda word: distance.euclidean(embeddings_index[word], embedding))
find_closest_embeddings((embedding_matrix[t.word_index['love']]))


# ## Task 5: Use the Word Embeddings to Fit a Neural Network for Classification

# First, prepare the Validation and Test data using the same tokenizer as was fit on the training data (so that we get the same indexes for the same vocabulary). This will also exclude any new words that show up in Validation / Test that weren't seen during training.

# In[32]:


encoded_val_doc = t.texts_to_sequences(val)
padded_val_doc = pad_sequences(encoded_val_doc, maxlen = max_length, padding = 'post')
encoded_test_doc = t.texts_to_sequences(test)
padded_test_doc = pad_sequences(encoded_test_doc, maxlen = max_length, padding = 'post')


# In[33]:


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


# In[34]:


history = model.fit(padded_docs, y_train, epochs = 10, verbose = 1, batch_size = 32, validation_data = (padded_val_doc, y_val))


# In[35]:


#save model:
# model.save_weights('imdb_glove_sequential_model.h5')


# In[36]:


acc = history.history['acc']
print ("Accuracy history: ",acc)
val_acc = history.history['val_acc']
print("\nValidation history: ",val_acc)
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[38]:


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


# In[40]:


pred_test = model.predict(padded_test_doc)
rounded_pred_test = np.round(pred_test)
cf_matrix = confusion_matrix(y_test, rounded_pred_test)
meplot.cf_matrix_heatmap(cf_matrix)


# In[41]:


print(classification_report(y_test,rounded_pred_test))


# ## Task 6: Model Explainability

# In[45]:


from lime import lime_text
from lime.lime_text import LimeTextExplainer

def predict_for_lime(text_array):
    encoded =t.texts_to_sequences(text_array)
    text_data = pad_sequences(encoded, maxlen=max_length,padding='post')
    pred=model.predict(text_data)
    return pred

# test the predicition function
print ("Verify if predictions are correct for the function")
print(predict_for_lime([train[0],train[533]]))
print(y_train[0], y_train[533])
#initilaize Lime for text
explainer = LimeTextExplainer(class_names=["Positive"])


# In[46]:


#Check explanation for a negative review
exp = explainer.explain_instance(str(train[22]), predict_for_lime, num_features=10, top_labels=1)
exp.show_in_notebook()

