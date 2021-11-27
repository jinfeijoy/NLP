# NLP
 
## MyNLPJourney
- My NLP learning plan and references.
- Summary of NLP Projects: https://github.com/jinfeijoy/NLP/blob/main/MyNLPJourney/NLP2021Plan.md

## Review Sentiment Analysis
* [kaggle_IMDB_Review](https://github.com/jinfeijoy/NLP/tree/main/kaggle_IMDB_Review) (2021May)
   * Kaggle IBDM Review task, 1st task to learn NLP basic. Create my own function to do text pre-processing. Tried traditional ML model and deeplearning word embedding. 
   * binary target, word embedding + RNN = 0.75 accu
   
## Twitter Sentiment Analysis
* [kaggle_Twitter_sentiment](https://github.com/jinfeijoy/NLP/tree/main/kaggle_Twitter_sentiment)
* [kaggle_Covid19_vaccine_Twitter](https://github.com/jinfeijoy/NLP/tree/main/kaggle_Covid19_vaccine_Twitter)
* Summary:
  * Traditional twitter sentiment analysis with same domain: 
    * binary classification: RNN (0.74), LSTM (0.816)
    * 3-class classification: RNN (.55), LSTM (.6807), AWD-LSTM (0.74), transformer distilbert (0.38)
    * RNN and LSTM (basic) are easy to understand and fast
    * AWD-LSTM was run with fastai, more complicated and time consuming 
  * twitter sentiment analysis with different domain: transferlearning with fastai (AWD-LSTM) 
    * tried other pre-trained model (distilbert), the performance is not good and model might be wrong, so for short sentence like tweets, if we have dataset can be trained and it won't take long time to train, it is better to train our own model with similar content/domain. For long sentence or article, we can discuss this later. 
  * Vaccine Tracker: visualization tools
 
 ## Document analysis (opinion mining)
   * [Kaggle Article Analysis](https://github.com/jinfeijoy/NLP/tree/main/kaggle_article_analysis)
     * article sentiment
     * article classification 

## [Topic Modelling and Text Summarization](https://github.com/jinfeijoy/NLP/tree/main/topic_modelling_text_summary)
   * Document Classification
     * doc2vec
     * transformers 
   * Topic Modelling
   * Text Summarization
     * Text Extraction: tf + similarity; word embedding + text rank
     * Text Abstraction: transformers 
   * Text Generation
     * fastai, gpt2, lstmrnn

## [Sementic Analysis](https://github.com/jinfeijoy/NLP/tree/main/semantic_analysis)
  * Keyword Extraction
  * Intent Classification

