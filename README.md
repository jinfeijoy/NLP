# NLP
 
## MyNLPJourney
My NLP learning plan and references.

## Review Sentiment Analysis
* kaggle_IMDB_Review()
   * Kaggle IBDM Review task, 1st task to learn NLP basic. Create my own function to do text pre-processing. Tried traditional ML model and deeplearning word embedding. (2021May)

## Twitter Sentiment Analysis
* kaggle_Twitter_sentiment()
* kaggle_Covid19_vaccine_Twitter()
* Summary:
  * Traditional twitter sentiment analysis with same domain: 
    * binary classification: RNN (0.74), LSTM (0.816)
    * 3-class classification: RNN (.55), LSTM (.6807), AWD-LSTM (0.74) 
    * RNN and LSTM (basic) are easy to understand and fast
    * AWD-LSTM was run with fastai, more complicated and time consuming 
  * twitter sentiment analysis with different domain: transferlearning with fastai (AWD-LSTM) 
    * tried other pre-trained model (BERT), the performance is not good, so for short sentence like tweets, if we have dataset can be trained and it won't take long time to train, it is better to train our own model with similar content/domain. For long sentence or article, we can discuss this later. 
