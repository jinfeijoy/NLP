# Twitter Sentiment Analysis

### Dataset 
  - [Kaggle Twitter Sentiment](https://www.kaggle.com/maxjon/complete-tweet-sentiment-extraction-data)
  - [Kaggle Twitter Sentiment 140 (no neutral sentiment)](https://www.kaggle.com/kazanova/sentiment140)
  - [Kaggle Covie Twitter Data](https://www.kaggle.com/arunavakrchakraborty/covid19-twitter-dataset)

- To do list:
  - sentiment analysis
    - word embedding with RNN (May 26) <Basic_twitter_sentiment_analysis>
    - contextual word embedding with fastai (May 29) <Basic_twitter_sentiment_analysis_fastai>
### Code Summary
- Local
  - Word Embedding (GloVe + RNN/LSTM): 
    - Binary Classification: [Basic_twitter_sentiment_analysis](http://localhost:8888/notebooks/Desktop/PersonalLearning/GitHub/NLP/kaggle_Twitter_sentiment/notebook/Basic_twitter_sentiment_analysis.ipynb)
    - 3 Classes Classification: [Basic_twitter_sentiment_analysis_smallDataset](http://localhost:8888/notebooks/Desktop/PersonalLearning/GitHub/NLP/kaggle_Twitter_sentiment/notebook/Basic_twitter_sentiment_analysis_smallDataset.ipynb)
- Google Colab
  - fastai + AWD_LSTM: [Basic_twitter_sentiment_analysis_fastai](http://localhost:8888/notebooks/Desktop/PersonalLearning/GitHub/NLP/kaggle_Twitter_sentiment/notebook/Basic_twitter_sentiment_analysis_fastai.ipynb): Fine-tune on tweets and predict unlabeled tweets



### Notebook
* Analyze_covid_twitter_after_fastai_prediction: summarize hashtag and sentiment 
* Basic_twitter_sentiment_analysis: applied RNN and LSTM to do binary classification
* Basic_twitter_sentiment_analysis_fastai: AWD_LSTM on google colab with labeled tweets data and unlabeled covid tweets dataset
* Basic_twitter_sentiment_analysis_fastai_py37: tried to install py37 and run it locally, but failed
* Basic_twitter_sentiment_analysis_smallDataset: did RNN and LSTM on smaller twitter dataset with 3-class target variable, to compare the performance with AWD_LSTM
* Load_model_and_do_prediction: load model file from google colab and do prediction
* Covid_Twitter_sentiment_analysis: word cloud and topic modelling
* Covid_twitter_emotion_analysis: generate emotion prediction dataset by using AWD_LSTM in google colab

### Other Python Package: tidytext
The tidytext package contains 3 general purpose lexicons in the sentiments dataset.

* AFINN - listing of english words rated for valence between -5 and +5
* bing - listing of positive and negative sentiment
* nrc - list of English words and their associations with 8 emotions (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust) and 2 sentiments (negative and positive); binary
