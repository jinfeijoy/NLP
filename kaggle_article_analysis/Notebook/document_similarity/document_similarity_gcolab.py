#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import io
import os
import re
from google.colab import drive
get_ipython().system('pip install transformers')
get_ipython().system('pip install pickle5')
from transformers import pipeline
drive.mount('/content/drive')


# In[2]:


data_path = '/content/drive/MyDrive/colab_data'
processed_letter_df = pd.read_csv(os.path.join(data_path, 'processed_letter.csv'))
processed_letter_df['tokens'] = processed_letter_df.clean_letter.apply(lambda x: x.split(' '))


# ## Transformers sentiment analysis

# In[3]:


from transformers import pipeline
classifier = pipeline('sentiment-analysis')


# In[8]:


top10freq_words = ['earnings', 'insurance', 'shareholder', 'stock', 'loss', 'gain', 'profit', 'increase', 'sell', 'rate']
df = pd.DataFrame(columns=['text', 'sentiment', 'score'])
for w in top10freq_words:
  result = classifier(w)
  new_row = {'text': w, 'sentiment':result[0]['label'], 'score':result[0]['score']}
  df = df.append(new_row, ignore_index=True)
df.head(3)


# ## Transformers QA

# In[14]:


import pickle5 as pickle
with open(os.path.join(data_path, 'letters_dict.pickle'), 'rb') as handle:
    letters_dict = pickle.load(handle)


# In[15]:


def get_answer_using_qa(nlp, question, context):
    """Get answer using a classifier trained with the QA technique
    Parameters
    ----------
    nlp: Pipeline
        Trained QA Pipeline
    question: String
        Question that the model will answer
    context: String
        The Context of the question
    Returns
    -------
    Tuple
        The answer, the score, the start position of the answer at the text and the final
        position of the answer at the text
    """
    result = nlp(question=question, context=context)

    return result['answer'], round(result['score'], 4), result['start'], result['end']


# In[17]:


nlp = pipeline("question-answering")


# In[18]:


print(sentence_after_regex_2008)


# In[26]:


year = 2020
sentence_after_regex = re.sub("[^a-z0-9A-Z\,]", " ", letters_dict[year])
answer, result, start, end = get_answer_using_qa(nlp, "what's the biggest profit this year?", sentence_after_regex)
print(f"Answer: '{answer}', score: {result}, start: {start}, end: {end}")


# ## Transformer embedding

# In[4]:


get_ipython().system('pip install sentence_transformers')
from sentence_transformers import SentenceTransformer


# In[ ]:


model = SentenceTransformer()

