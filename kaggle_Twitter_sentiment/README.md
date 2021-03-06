# Twitter Sentiment Analysis
- Dataset can be found here: 
  - https://www.kaggle.com/maxjon/complete-tweet-sentiment-extraction-data
  - https://www.kaggle.com/kazanova/sentiment140 this dataset dont have neutral sentiment

- To do list:
  - sentiment analysis
    - word embedding with RNN (May 26) <Basic_twitter_sentiment_analysis>
    - contextual word embedding with fastai (May 29) <Basic_twitter_sentiment_analysis_fastai>


# Covid Twitter Sentiment Analysis (with fastai)
- analyze convid tweet
  - https://www.kaggle.com/arunavakrchakraborty/covid19-twitter-dataset this dataset include covid tweet

# Notebook
* Analyze_covid_twitter_after_fastai_prediction: summarize hashtag and sentiment 
* Basic_twitter_sentiment_analysis: applied RNN and LSTM to do binary classification
* Basic_twitter_sentiment_analysis_fastai: AWD_LSTM on google colab with labeled tweets data and unlabeled covid tweets dataset
* Basic_twitter_sentiment_analysis_fastai_py37: tried to install py37 and run it locally, but failed
* Basic_twitter_sentiment_analysis_smallDataset: did RNN and LSTM on smaller twitter dataset with 3-class target variable, to compare the performance with AWD_LSTM
* Load_model_and_do_prediction: load model file from google colab and do prediction
* Covid_Twitter_sentiment_analysis: word cloud and topic modelling
* Covid_twitter_emotion_analysis: generate emotion prediction dataset by using AWD_LSTM in google colab
