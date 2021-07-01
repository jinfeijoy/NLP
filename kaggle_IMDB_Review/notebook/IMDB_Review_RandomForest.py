#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
root_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data'


# ## Load Data

# In[3]:


raw_data = pd.read_csv(os.path.join(root_path, "IMDB_Dataset.csv"))
data = raw_data[raw_data.review.isnull()==False]
data['label'] = np.where(data['sentiment']=='positive', 1,0)
data = data.drop_duplicates()
data.insert(0, 'index', data.index + 1)


# In[3]:


data.head(3)


# ## Data Visualization

# In[4]:


for idx, row in raw_data.sample(5).iterrows():
    print("review:")
    print(row['review'])
    print()
    print("sentiment:")
    print(row['sentiment'])
    print("================================================================================================================")


# ## Explore Data

# In[5]:


data.groupby('sentiment').agg('count')


# In[6]:


print('maximum length of reviews:', DataExploration.text_length_summary(data, 'review', 'max'))
print('minimal length of reviews:', DataExploration.text_length_summary(data, 'review', 'min'))
print('average length of reviews:', DataExploration.text_length_summary(data, 'review', 'avg'))
print('median length of reviews:', DataExploration.text_length_summary(data, 'review', 'median'))


# In[4]:


positive_review = data[data['sentiment']=='positive']
negative_review = data[data['sentiment']=='negative']
preprocessed_tokens = textClean.pipeline(data['review'][0:500].to_list(), multi_gram = [1], lower_case=True, deacc=False, encoding='utf8',
                                           errors='strict', stem_lemma = 'lemma', tag_drop = ['V'], nltk_stop=True, stop_word_list=['movie','film','movies','films'], 
                                           check_numbers=True, word_length=3, remove_consecutives=True)
pos_tokens = textClean.pipeline(positive_review['review'][0:500].to_list(), multi_gram = [1], lower_case=True, deacc=False, encoding='utf8',
                                           errors='strict', stem_lemma = 'lemma', tag_drop = ['V'], nltk_stop=True, stop_word_list=['movie','film','movies','films'], 
                                           check_numbers=True, word_length=3, remove_consecutives=True)
neg_tokens = textClean.pipeline(negative_review['review'][0:500].to_list(), multi_gram = [1], lower_case=True, deacc=False, encoding='utf8',
                                           errors='strict', stem_lemma = 'lemma', tag_drop = ['V'], nltk_stop=True, stop_word_list=['movie','film','movies','films'], 
                                           check_numbers=True, word_length=3, remove_consecutives=True)
pos_tokens[0]


# In[41]:


postop10tfidf = tfidf.get_top_n_tfidf_bow(pos_tokens, top_n_tokens = 30)
negtop10tfidf = tfidf.get_top_n_tfidf_bow(neg_tokens, top_n_tokens = 30)
print('top 10 negative review tfidf', negtop10tfidf)
print('top 10 positive review tfidf', postop10tfidf)


# In[36]:


top10_freq_list = DataExploration.get_topn_freq_bow(preprocessed_tokens, topn = 10)
top10_posfreq_list = DataExploration.get_topn_freq_bow(pos_tokens, topn = 10)
top10_negfreq_list = DataExploration.get_topn_freq_bow(neg_tokens, topn = 10)
print(top10_freq_list)
print(top10_posfreq_list)
print(top10_negfreq_list)


# In[37]:


DataExploration.generate_word_cloud(pos_tokens)


# In[38]:


DataExploration.generate_word_cloud(neg_tokens)


# ## Model Development

# ### 1. Split Dataset

# In[3]:


preprocessed_tokens = textClean.pipeline(data['review'][0:100].to_list(), multi_gram = [1], lower_case=True, deacc=False, encoding='utf8',
                                           errors='strict', stem_lemma = 'lemma', tag_drop = [], nltk_stop=True, stop_word_list=['movie'], 
                                           check_numbers=False, word_length=3, remove_consecutives=True)

dictionary = DocVector.generate_corpus_dict(preprocessed_tokens, no_below =1,
                                            no_above = 0.5, keep_n = 100000)
bow_corpus = DocVector.create_document_vector(preprocessed_tokens, dictionary)
my_df = DocVector.get_vocab_matrix(bow_corpus, dictionary)
my_df.head(3)


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(my_df.loc[:99,:], data['label'][0:100], test_size = 0.33, random_state = 11)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 11)


# ### 2. Feature Engineering
# * Apart from vectorization process, addtional features can be created from the dataset such as:
#     * Length of comments 
#     * Number of entities (using Named-Entity Recognition(NER))
#     * One-hot encoding of entities
#     * Number of positive and negative words (not only meant for sentiment analysis but useful in general)
#     * Number of special characters
#     * One-hot encoding of special characters
#     * Part-of-speech(POS) tags
# * Techniques for feature selection to explore:
#     * SelectKBest: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
#     * Recursive Feature Elimination (RFE): https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html

# ### 3. Preliminary Result

# #### Training the model

# In[5]:


sentiment_analyzer = RandomForestClassifier()
sentiment_analyzer.fit(X_train, y_train)


# #### Evaluating the model using 5-Fold CrossValidation

# In[15]:


five_fold_accuracy = cross_val_score(sentiment_analyzer, X_train, y_train, cv = 5)
five_fold_accuracy


# ### 4. Hyperparameter Tuning and New Result
# link: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# 
# random forest hypterparameter tuning: https://www.topcoder.com/thrive/articles/understanding-random-forest-and-hyper-parameter-tuning

# In[16]:


from sklearn.model_selection import GridSearchCV
grid_parameters = {
    'bootstrap': [True, False],
    'max_depth': [10, 20],# 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600]#, 800, 1000, 1200, 1400, 1600, 1800, 2000]
}


# In[18]:


# model_param_search = GridSearchCV(sentiment_analyzer, param_grid = grid_parameters, scoring = 'accuracy', cv = 3)
# model_param_search.fit(X_train, y_train)
# model_param_search.best_params_


# In[16]:


sentiment_analyzer = RandomForestClassifier(bootstrap = True, max_depth = 20, max_features = 'sqrt', min_samples_leaf = 2, min_samples_split = 5, n_estimators = 200)
sentiment_analyzer.fit(X_train, y_train)


# In[17]:


five_fold_accuracy = cross_val_score(sentiment_analyzer, X_train, y_train, cv = 5)
five_fold_accuracy


# #### Addtional Evaluation Metrics

# Classification Report

# In[18]:


predictions = sentiment_analyzer.predict(X_val)
print(classification_report(y_val, predictions))


# Confusion Matrix

# In[19]:


cf_matrix = confusion_matrix(y_val, predictions)
meplot.cf_matrix_heatmap(cf_matrix)


# In[20]:


meplot.plot_randomfp_roc(X_train, y_train, X_val, y_val, sentiment_analyzer.predict_proba(X_val)[:,1], model_label = 'RandomForest')    


# Reeiver Operating Characteristic (ROC) Curve

# #### SHAP

# In[6]:


plot, shap_val = meshap.get_tree_SHAP_plot(model = sentiment_analyzer, X = X_val, define_index = False, index_value = 1)
plot


# In[7]:


shap_val.head(10)


# In[8]:


import shap
shap_explainer = shap.TreeExplainer(sentiment_analyzer,X_val)
shap_values = shap_explainer.shap_values(X_val)


shap.summary_plot(shap_values, X_val)


# In[23]:


shap.initjs()
def shap_plot(j):
    explainerModel = shap.TreeExplainer(sentiment_analyzer)
    shap_values_Model = explainerModel.shap_values(X_val)
    p = shap.force_plot(explainerModel.expected_value[0], shap_values_Model[0][j], X_val.iloc[[j]])
    print("Review:",data.review[X_val.iloc[[j]].index[0]])
    return(p)


# In[24]:


shap_plot(0)


# In[47]:


sentiment_analyzer.feature_importances_


# In[48]:


meplot.plot_var_imp(sentiment_analyzer.feature_importances_, X_train.columns, 20)


# In[27]:


meplot.plot_var_imp_both_side(X_train, y_train, X_train.columns, data, sentiment_analyzer.feature_importances_, topn = 40)


# ## Pipelines

# #### Define Transformer and Pipeline

# In[9]:


from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
class TextCleaning(BaseEstimator):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        preprocessed_tokens = textClean.pipeline(X.to_list(), multi_gram = [1], lower_case=True, deacc=False, encoding='utf8',
                                                 errors='strict', stem_lemma = 'lemma', tag_drop = [], nltk_stop=True, 
                                                 stop_word_list=[], check_numbers=False, word_length=3,
                                                 remove_consecutives=True)
        
        return preprocessed_tokens
                                      
class TextVectors(BaseEstimator):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
                                    
    def transform(self, X, y=None):

        bow_corpus = DocVector.create_document_vector(X, dictionary)
        my_df = DocVector.get_vocab_matrix(bow_corpus, dictionary)
        return my_df
    

my_pipeline = Pipeline(steps = [
    ('Text Cleaning', TextCleaning()),
    ('Vectorization', TextVectors()),
    ('Prediction', RandomForestClassifier())
])   


# #### Access Individual Steps

# In[23]:


# processed_doc = my_pipeline['Text Cleaning'].transform(data['review'][0:5000])
# dictionary = DocVector.generate_corpus_dict(processed_doc, no_below =1, no_above = 0.5, keep_n = 100000)
# dictionary.save(os.path.join(root_path,'dictionary.gensim'))


# In[10]:


dictionary = corpora.Dictionary.load(os.path.join(root_path,'dictionary.gensim'))


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(data['review'][0:2000], data['label'][0:2000], test_size = 0.33, random_state = 11)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 11)


# In[12]:


my_pipeline.fit(X_train, list(y_train))


# In[13]:


predictions = my_pipeline.predict(X_val)


# #### Model Performance

# In[14]:


print(classification_report(y_val, predictions))
cf_matrix = confusion_matrix(y_val, predictions)
meplot.cf_matrix_heatmap(cf_matrix)


# #### Feature Importance

# In[15]:


my_rf_model = my_pipeline['Prediction']
sample_process_doc = my_pipeline['Text Cleaning'].transform(data['review'][0:500])
sample_vec = my_pipeline['Vectorization'].transform(sample_process_doc)
sample_label = list(data['label'])[:500]


# In[16]:


meplot.plot_var_imp(my_rf_model.feature_importances_, sample_vec.columns, 20)


# In[17]:


meplot.plot_var_imp_both_side(sample_vec, sample_label, sample_vec.columns, data, my_rf_model.feature_importances_, topn = 40)

